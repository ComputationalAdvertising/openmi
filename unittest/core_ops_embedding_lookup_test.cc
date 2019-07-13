#include "tensor_util.h"
#include "tensor_types.h" 
using namespace openmi;

/*!
 * \brief SegmentSumOpImpl 
 *     in: m * embedding_size (m = sum(offset))
 * offset: batch_size * 1
 *    out: batch_size * embedding_size
 */
template <typename T, int NDIMS = 2>
void SegmentSumOpImpl(typename TTypes<T>::Matrix in, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix out) {
  Eigen::array<Eigen::DenseIndex, 1> axis({{0}});
  int embedding_size = in.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    start_idx[0] = start_idx[0] + extent_idx[0];
    extent_idx[0] = offset(i);
    start[0] = i;
    if (offset(i) == 1) {
      out.slice(start, extent) = in.slice(start_idx, extent_idx);
    } else {
      out.slice(start, extent) = in.slice(start_idx, extent_idx).sum(axis).eval().reshape(reshape_dim);
    }
    DLOG(INFO) << "index:" << i << ", sum pooling:\n" << out.slice(start, extent);
  }
  DLOG(INFO) << "out:\n" << out;
}

/*!
 * \brief SegmentSumGradOpImpl
 *      dY: batch_size * embedding_size
 *  offset: batch_size * 1
 *      dX: m * embedding_size. m = sum(offset)
 */
template <typename T, int NDIMS = 2> 
void SegmentSumGradOpImpl(typename TTypes<T>::Matrix dY, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix dX) {
  int embedding_size= dY.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> bcast_dims({{1,1}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i);
    start_idx[0] = i;
    start[0] = start[0] + extent[0];
    extent[0] = value_size;
    if (value_size == 1) {
      dX.slice(start, extent) = dY.slice(start_idx, extent_idx);
      continue;
    }
    bcast_dims[0] = value_size;
    dX.slice(start, extent) = dY.slice(start_idx, extent_idx).broadcast(bcast_dims).eval();
    DLOG(INFO) << "index: " << i << ", dX:\n" << dX.slice(start, extent);
  }
  DLOG(INFO) << "SegmentSumGradOp dX:\n" << dX;
}

/*!
 * \brief SegmentMeanOpImpl 
 *     in: m * embedding_size (m = sum(offset))
 * offset: batch_size * 1
 *    out: batch_size * embedding_size
 */
template <typename T, int NDIMS = 2>
void SegmentMeanOpImpl(typename TTypes<T>::Matrix in, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix out) {
  Eigen::array<Eigen::DenseIndex, 1> axis({{0}});
  int embedding_size = in.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    start_idx[0] = start_idx[0] + extent_idx[0];
    extent_idx[0] = offset(i);
    start[0] = i;
    if (offset(i) == 1) {
      out.slice(start, extent) = in.slice(start_idx, extent_idx);
    } else {
      out.slice(start, extent) = in.slice(start_idx, extent_idx).mean(axis).eval().reshape(reshape_dim);
    }
    DLOG(INFO) << "index:" << i << ", mean pooling:\n" << out.slice(start, extent);
  }
  DLOG(INFO) << "SegmentMeanOpImpl out:\n" << out;
}

/*!
 * \brief SegmentMeanGradOpImpl
 *      dY: batch_size * embedding_size
 *  offset: batch_size * 1
 *      dX: m * embedding_size. m = sum(offset)
 */
template <typename T, int NDIMS = 2> 
void SegmentMeanGradOpImpl(typename TTypes<T>::Matrix dY, typename TTypes<int32_t>::Vector offset, typename TTypes<T>::Matrix dX) {
  int embedding_size= dY.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> bcast_dims({{1,1}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    int value_size = offset(i);
    start_idx[0] = i;
    start[0] = start[0] + extent[0];
    extent[0] = value_size;
    if (value_size == 1) {
      dX.slice(start, extent) = dY.slice(start_idx, extent_idx);
      continue;
    }
    bcast_dims[0] = value_size;
    dX.slice(start, extent) = (1.0 / value_size) * dY.slice(start_idx, extent_idx).broadcast(bcast_dims).eval();
    DLOG(INFO) << "index: " << i << ", dX:\n" << dX.slice(start, extent);
  }
  DLOG(INFO) << "SegmentMeanGradOp dX:\n" << dX;
}


/*!
 * \brief EmbeddingLookupOpImpl
 *        W: m * embdding_size
 *        X: m * 1
 *   offset: batch_size * 1
 *      out: batch_size * embedding_size
 */
template <typename T, int NDIMS = 2>
void EmbeddingLookupOpImpl(typename TTypes<T>::Matrix W, 
                           typename TTypes<T>::Matrix X, 
                           typename TTypes<int32_t>::Vector offset, 
                           typename TTypes<T>::Matrix out) {
  int embedding_size = W.dimension(1);
  Eigen::array<Eigen::DenseIndex, 2> bcast_dims({{1, embedding_size}});
  auto WX = W * X.broadcast(bcast_dims);
  
  Eigen::array<Eigen::DenseIndex, 1> axis({{0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start_idx({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent_idx({{0,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> start({{0,0}});
  Eigen::array<Eigen::DenseIndex, NDIMS> extent({{1,embedding_size}});
  Eigen::array<Eigen::DenseIndex, NDIMS> reshape_dim({{1,embedding_size}});

  for (auto i = 0; i < offset.dimension(0); ++i) {
    start_idx[0] = start_idx[0] + extent_idx[0];
    extent_idx[0] = offset(i);
    auto segment = WX.slice(start_idx, extent_idx);
    auto sum_pooling = segment.sum(axis).eval().reshape(reshape_dim);
    start[0] = i;
    out.slice(start, extent) = sum_pooling;
    DLOG(INFO) << "index:" << i << ", sum pooling:\n" << sum_pooling;
  }
  DLOG(INFO) << "out:\n" << out;
}

int main(int argc, char** argv) {
  typedef float T;
  const int NDIMS = 2;
  std::string w_shape("20,2");
  Tensor& w = test::CreateTensor(w_shape, NDIMS, DT_FLOAT);
  w.tensor<float, NDIMS>().setConstant(2.0);
  w.tensor<float, NDIMS>()(0,0) = 0.0;
  w.tensor<float, NDIMS>()(1,0) = 1.0;
  w.tensor<float, NDIMS>()(2,0) = 2.0;
  w.tensor<float, NDIMS>()(3,0) = 3.0;
  w.tensor<float, NDIMS>()(4,0) = 4.0;
  LOG(INFO) << "w:\n" << w.tensor<float, NDIMS>() << ", size: " << w.shape().NumElements();

  std::string x_shape("20");
  Tensor& x = test::CreateTensor(x_shape, NDIMS, DT_FLOAT);
  x.tensor<float, NDIMS>().setConstant(2.4);
  LOG(INFO) << "x:\n" << x.tensor<float, 1>() << ", size: " << x.shape().NumElements();

  std::string offset_shape("10");
  Tensor& row_offset = test::CreateTensor(offset_shape, 1, DT_INT32);
  row_offset.tensor<int32_t, 1>().setConstant(1);
  row_offset.tensor<int32_t, 1>()(0) = 3;
  row_offset.tensor<int32_t, 1>()(1) = 2;
  row_offset.tensor<int32_t, 1>()(2) = 2;
  row_offset.tensor<int32_t, 1>()(9) = 7;
  LOG(INFO) << "row_offset:\n" << row_offset.tensor<int32_t, 1>() << ", size: " << row_offset.shape().NumElements(); 

  std::string result_shape("10,2");
  Tensor& result = test::CreateTensor(result_shape, NDIMS, DT_FLOAT);
  result.tensor<float, NDIMS>().setConstant(0);

  auto W = w.matrix<T>();
  auto X = x.matrix<T>();
  auto offset = row_offset.vec<int32_t>();
  auto out = result.matrix<T>();
  LOG(INFO) << "out:\n" << out;

  Tensor& wx = test::CreateTensor(w_shape, NDIMS, DT_FLOAT);
  int embedding_size = W.dimension(1);
  Eigen::array<Eigen::DenseIndex, NDIMS> bcast_dims({{1, embedding_size}});
  auto WX = wx.matrix<T>();
  WX = W * X.broadcast(bcast_dims);
  LOG(INFO) << "WX:\n" << WX;
  //SegmentSumOpImpl<T, NDIMS>(WX, offset, out);
  SegmentMeanOpImpl<T, NDIMS>(WX, offset, out);
  LOG(INFO) << "result.matrix<T>:\n" << result.matrix<T>();

  auto dw_node = test::CreateTensor(w_shape, NDIMS, DT_FLOAT);
  auto dW = dw_node.matrix<T>();
  // SegmentSumGradOpImpl<T, NDIMS>(out, offset, dW);
  SegmentMeanGradOpImpl<T, NDIMS>(out, offset, dW);
  LOG(INFO) << "dw_node.matrix<T>:\n" << dw_node.matrix<T>();

  //EmbeddingLookupOpImpl<T, NDIMS>(W, X, offset, out);

  return 0;
}
