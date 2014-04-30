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

#ifndef BFS_H_
#define BFS_H_

#include <GASengine/csr_problem.cuh>

struct bfs
{
  typedef int DataType;
  typedef DataType MiscType;
  typedef DataType GatherType;
  typedef int VertexId;
  typedef int SizeT;

  static const DataType INIT_VALUE = -1;
  static const bool allow_duplicates = true;

  struct VertexType
  {
    int* d_labels;
    int nodes;
    int edges;

    VertexType() :
        d_labels(NULL), nodes(0), edges(0)
    {
    }
  };

  struct EdgeType
  {
    int nodes; // #of nodes.
    int edges; // #of edges.

    EdgeType() :
        nodes(0), edges(0)
    {
    }
  };

  static void Initialize(const int directed, const int nodes, const int edges, int num_srcs,
      int* srcs, int* d_row_offsets, int* d_column_indices, int* d_column_offsets, int* d_row_indices, int* d_edge_values,
      VertexType &vertex_list, EdgeType &edge_list, int* d_frontier_keys[3],
      MiscType* d_frontier_values[3])
  {
    vertex_list.nodes = nodes;
    vertex_list.edges = edges;
    DataType* h_init_labels = new DataType[nodes];

    for (int i = 0; i < nodes; i++)
    {
      h_init_labels[i] = -1;
    }
    for (int i = 0; i < num_srcs; i++)
    {
      h_init_labels[srcs[i]] = 0;
    }

    b40c::util::B40CPerror(cudaMalloc((void**) &vertex_list.d_labels, nodes * sizeof(int)), "cudaMalloc VertexType::d_labels failed", __FILE__, __LINE__);

    // Initialize d_labels
    if (b40c::util::B40CPerror(
        cudaMemcpy(vertex_list.d_labels, h_init_labels, nodes * sizeof(DataType),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_labels failed", __FILE__,
        __LINE__))
      exit(0);

    delete[] h_init_labels;

    printf("Starting vertex: ");
    for (int i = 0; i < num_srcs; i++)
      printf("%d ", srcs[i]);
    printf("\n");

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[0], srcs, num_srcs * sizeof(int),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_keys[1], srcs, num_srcs * sizeof(int),
            cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_keys failed", __FILE__,
        __LINE__))
      exit(0);

    int init_value[1] = { 0 };
    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[0], init_value,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

    if (b40c::util::B40CPerror(
        cudaMemcpy(d_frontier_values[1], init_value,
            num_srcs * sizeof(int), cudaMemcpyHostToDevice),
        "CsrProblem cudaMemcpy d_frontier_values failed", __FILE__,
        __LINE__))
      exit(0);

  }

  static SrcVertex srcVertex()
  {
    return SINGLE;
  }

  static GatherEdges gatherOverEdges()
  {
    return NO_GATHER_EDGES;
  }

  static ApplyVertices applyOverEdges()
  {
    return NO_APPLY_VERTICES;
  }

  static ExpandEdges expandOverEdges()
  {
    return EXPAND_OUT_EDGES;
  }

  static PostApplyVertices postApplyOverEdges()
  {
    return NO_POST_APPLY_VERTICES;
  }

  /**
   * the binary operator
   */
  struct gather_sum
  {
    __device__
    GatherType operator()(GatherType left, GatherType right)
    {
      return left + right;
    }
  };

  /**
   * For each vertex in the frontier,
   */
  struct gather_vertex
  {
    __device__
    void operator()(const int vertex_id, const GatherType final_value,
        VertexType &vertex_list, EdgeType &edge_list)
    {

    }
  };

  struct gather_edge
  {
    __device__
    void operator()(const int vertex_id, const int edge_id, const int neighbor_id_in,
        VertexType &vertex_list, EdgeType &edge_list, GatherType& new_value)
    {

    }
  };

  struct apply
  {
    __device__
    void operator()(const int vertex_id, const int iteration, GatherType gathervalue,
        VertexType& vertex_list, EdgeType& edge_list, char& changed)
    {
      changed = 1;
    }
  };

  /** post-apply function (invoked after threads in apply() synchronize at a memory barrier). */
  struct post_apply
  {
    __device__
    void operator()(const int vertex_id, VertexType& vertex_list, EdgeType& edge_list, GatherType* gather_tmp)
    {
    }
  };

  struct expand_vertex
  {
    __device__
    bool operator()(const int vertex_id, const char changed, VertexType &vertex_list, EdgeType& edge_list)
    {
      return true;
    }
  };

  struct expand_edge
  {
    __device__
    void operator()(const bool changed, const int iteration,
        const int vertex_id, const int neighbor_id_in, const int edge_id,
        VertexType& vertex_list, EdgeType& edge_list, int& frontier, int& misc_value)
    {
//      misc_value = vertex_id;
//      if(neighbor_id_in == 4120 || neighbor_id_in == 4480 || vertex_id == 4096)
//        printf("Expand: vertex_id=%d, neighbor_id_in = %d\n", vertex_id, neighbor_id_in);
      frontier = neighbor_id_in;
    }
  };

  struct contract
  {
    __device__
    void operator()(const int iteration, char *d_bitmap_visited, int &vertex_id,
        VertexType &vertex_list, EdgeType &edge_list, GatherType* gather_tmp, int& misc_value)
    {
      int row_id = vertex_id;
      int byte_id = row_id / 8;
      int bit_off = row_id % 8;
      char mask = 1 << bit_off;
      char is_visited = d_bitmap_visited[byte_id] & mask;
//      if(row_id == 4120 || row_id == 4480)
//          printf("row_id=%d, mask=%d, is_visited=%d\n", row_id, mask, is_visited);
      if(is_visited != 0 )
        vertex_id = -1;
//      // Load label of node
//      int row_id = vertex_id;
//      int label;
//      label = vertex_list.d_labels[row_id];
////      printf("row_id=%d, label=%d, iteration=%d\n", row_id, label, iteration);
//
//      if (label != -1)
//      {
//
//        // Seen it
//        vertex_id = -1;
//
//      }
//      else
//      {
//
//        // Update label with current iteration
//        vertex_list.d_labels[row_id] = iteration + 1;
//      }
    }
  };

  static void extractResult(VertexType& vertex_list, DataType* h_output)
  {
    cudaMemcpy(h_output, vertex_list.d_labels, sizeof(DataType) * vertex_list.nodes, cudaMemcpyDeviceToHost);
  }

};

#endif /* BFS_H_ */
