/**
Copyright 2013-2014 SYSTAP, LLC.  http://www.systap.com

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

   http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.

This work was (partially) funded by the DARPA XDATA program under
AFRL Contract #FA8750-13-C-0002.

This material is based upon work supported by the Defense Advanced
Research Projects Agency (DARPA) under Contract No. D14PC00029.
*/

#ifndef UTIL_H_
#define UTIL_H_

#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <thrust/fill.h>
#include <thrust/extrema.h>
#include <thrust/binary_search.h>
#include <thrust/transform.h>
#include <thrust/gather.h>
#include <thrust/scatter.h>
#include <thrust/sequence.h>
#include <thrust/scan.h>
#include <thrust/sort.h>

template<typename IndexType>
struct empty_row_functor
{
  typedef bool result_type;

  template<typename Tuple>
  __host__ __device__
  bool operator()(const Tuple& t) const
      {
    const IndexType a = thrust::get < 0 > (t);
    const IndexType b = thrust::get < 1 > (t);

    return a != b;
  }
};

template<typename OffsetIter, typename IndexIter>
void offsets_to_indices(const size_t n, const size_t m, const OffsetIter offsets, IndexIter indices)
{
  typedef typename OffsetIter::value_type OffsetType;

  // convert compressed row offsets into uncompressed row indices
  thrust::fill(indices, indices + m, OffsetType(0));
  thrust::scatter_if(thrust::counting_iterator < OffsetType > (0),
      thrust::counting_iterator < OffsetType > (n - 1),
      offsets,
      thrust::make_transform_iterator(
          thrust::make_zip_iterator(thrust::make_tuple(offsets, offsets + 1)),
          empty_row_functor<OffsetType>()),
      indices);
  thrust::inclusive_scan(indices, indices + m, indices, thrust::maximum<OffsetType>());
}

template<typename OffsetIter, typename IndexIter>
void indices_to_offsets(const size_t n, const size_t m, const IndexIter indices, OffsetIter offsets)
{
  typedef typename OffsetIter::value_type OffsetType;

  // convert uncompressed row indices into compressed row offsets
  thrust::lower_bound(indices,
      indices + m,
      thrust::counting_iterator < OffsetType > (0),
      thrust::counting_iterator < OffsetType > (n),
      offsets);
}

template<typename Array1, typename Array2, typename Array3, typename Array4>
void sort_by_column(const size_t indices_size, const size_t offsets_size, const Array1 columns, Array2 rows, Array3 column_offsets, Array4 permutation)
{
  typedef typename Array1::value_type IndexType;

  size_t N = indices_size;

//  printf("rows: ");
//  for (int i = 0; i < indices_size; i++)
//    std::cout << rows[i] << " ";
//  printf("\n");
//
//  printf("columns: ");
//  for (int i = 0; i < indices_size; i++)
//    std::cout << columns[i] << " ";
//  printf("\n");

  thrust::sequence(permutation, permutation + N);
  thrust::device_vector<IndexType> temp(columns, columns + N);
  thrust::stable_sort_by_key(temp.begin(), temp.end(), permutation);
  indices_to_offsets(offsets_size, N, temp.begin(), column_offsets);

//  printf("temp: ");
//  for (int i = 0; i < indices_size; i++)
//    std::cout << temp[i] << " ";
//  printf("\n");
//
//  printf("permutation: ");
//  for (int i = 0; i < indices_size; i++)
//    std::cout << permutation[i] << " ";
//  printf("\n");

  thrust::copy(rows, rows + N, temp.begin());
  thrust::gather(permutation, permutation + N, temp.begin(), rows);

//  printf("rows after: ");
//  for (int i = 0; i < indices_size; i++)
//    std::cout << rows[i] << " ";
//  printf("\n");


}

#endif /* UTIL_H_ */
