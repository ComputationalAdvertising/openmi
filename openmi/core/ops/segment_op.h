#ifndef OPENMI_CORE_OPS_SEGMENT_OP_H_
#define OPENMI_CORE_OPS_SEGMENT_OP_H_ 

#include "numeric_op.h"
#include "op_registry.h"

namespace openmi {

/*!
 * \brief SegmentSumOpImpl 
 *     in: m * embedding_size (m = sum(offset))
 * offset: batch_size * 1
 *    out: batch_size * embedding_size
 */

template <typename T, int NDIMS = 2>
struct SegmentSumOpImpl {
static void Compute(typename TTypes<T>::Matrix in, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix out) {
  Eigen::array<Eigen::DenseIndex, 1> axis({{0}});
  int embedding_size = in.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});
  int prev_offset = 0;

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i) - prev_offset;
    prev_offset = offset(i);
    extent_idx[0] = value_size;
    start[0] = i;
    if (value_size == 1) {
      out.slice(start, extent) = in.slice(start_idx, extent_idx);
    } else {
      out.slice(start, extent) = in.slice(start_idx, extent_idx).sum(axis).eval().reshape(reshape_dim);
    }
    start_idx[0] = start_idx[0] + extent_idx[0];
    DLOG(INFO) << "index:" << i << ", value_size:" << value_size << ", sum pooling:\n" << out.slice(start, extent);
  }
  DLOG(INFO) << "out:\n" << out;
}
}; // struct SegmentSumOpImpl

/*!
 * \brief SegmentSumGradOpImpl
 *      dY: batch_size * embedding_size
 *  offset: batch_size * 1
 *      dX: m * embedding_size. m = sum(offset)
 */
template <typename T, int NDIMS = 2> 
struct SegmentSumGradOpImpl {
static void Compute(typename TTypes<T>::Matrix dY, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix dX) {
  int embedding_size= dY.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> bcast_dim({{1,1}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});
  int32_t prev_offset = 0;

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i) - prev_offset;
    prev_offset = offset(i);
    start_idx[0] = i;
    start[0] = start[0] + extent[0];
    extent[0] = value_size;
    if (value_size == 1) {
      dX.slice(start, extent) = dY.slice(start_idx, extent_idx);
      continue;
    }
    DLOG(INFO) << "i:" << i << ",value_size:" << value_size;
    bcast_dim[0] = value_size;
    reshape_dim[0] = value_size;
    dX.slice(start, extent) = dY.slice(start_idx, extent_idx).broadcast(bcast_dim).eval().reshape(reshape_dim);
    DLOG(INFO) << "index:" << i << ", extent:" << extent[0] << ", dX:\n" << dX.slice(start, extent);
  } 
  DLOG(INFO) << __FUNCTION__ << " dX:\n" << dX;
}
}; // struct SegmentSumGradOpImpl

/*!
 * \brief SegmentMeanOpImpl 
 *     in: m * embedding_size (m = sum(offset))
 * offset: batch_size * 1
 *    out: batch_size * embedding_size
 */
template <typename T, int NDIMS = 2>
struct SegmentMeanOpImpl {
static void Compute(typename TTypes<T>::Matrix in, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix out) {
  Eigen::array<Eigen::DenseIndex, 1> axis({{0}});
  int embedding_size = in.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i) - start_idx[0];
    extent_idx[0] = value_size;
    start[0] = i;
    if (value_size == 1) {
      out.slice(start, extent) = in.slice(start_idx, extent_idx);
    } else {
      out.slice(start, extent) = in.slice(start_idx, extent_idx).mean(axis).eval().reshape(reshape_dim);
    }
    start_idx[0] = start_idx[0] + extent_idx[0];
    DLOG(INFO) << "index:" << i << ", value_size:" << value_size << ", mean pooling:\n" << out.slice(start, extent);
  }
  DLOG(INFO) << __FUNCTION__ << " out:\n" << out;
}
}; // struct SegmentMeanOpImpl

/*!
 * \brief SegmentMeanGradOpImpl
 *      dY: batch_size * embedding_size
 *  offset: batch_size * 1
 *      dX: m * embedding_size. m = sum(offset)
 */
template <typename T, int NDIMS = 2> 
struct SegmentMeanGradOpImpl {
static void Compute(typename TTypes<T>::Matrix dY, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix dX) {
  int embedding_size= dY.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> bcast_dims({{1,1}});
  int32_t prev_offset = 0;

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i) - prev_offset;
    prev_offset = offset(i);
    start_idx[0] = i;
    start[0] = start[0] + extent[0];
    extent[0] = value_size;
    if (value_size == 1) {
      dX.slice(start, extent) = dY.slice(start_idx, extent_idx);
      continue;
    }
    bcast_dims[0] = value_size;
    dX.slice(start, extent) = 1.0 / value_size * dY.slice(start_idx, extent_idx).broadcast(bcast_dims).eval();
    DLOG(INFO) << "index: " << i << ", dX:\n" << dX.slice(start, extent);
  } 
  DLOG(INFO) << __FUNCTION__ << " dX:\n" << dX;
}
}; // struct SegmentMeanGradOpImpl

} // namespace openmi
#endif // OPENMI_CORE_OPS_SEGMENT_OP_H_
