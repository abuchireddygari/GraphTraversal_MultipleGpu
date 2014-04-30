/******************************************************************************
 * 
 * Copyright 2010-2012 Duane Merrill
 * 
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 * 
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License. 
 * 
 * For more information, see our Google Code project site: 
 * http://code.google.com/p/back40computing/
 * 
 ******************************************************************************/

/*THIS FILE HAS BEEN MODIFIED FROM THE ORIGINAL*/

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

/******************************************************************************
 * GPU CSR storage management structure for problem data
 ******************************************************************************/

#pragma once

#include <b40c/util/basic_utils.cuh>
#include <b40c/util/cuda_properties.cuh>
#include <b40c/util/memset_kernel.cuh>
#include <b40c/util/cta_work_progress.cuh>
#include <b40c/util/error_utils.cuh>
#include <b40c/util/multiple_buffering.cuh>
#include <GASengine/problem_type.cuh>
#include <config.h>
#include <omp.h>
#include <util.cuh>
#include <thrust/device_vector.h>
#include <thrust/device_ptr.h>
#include <thrust/copy.h>
#include <vector>

using namespace b40c;

enum SrcVertex
{
  SINGLE, ALL
};

enum GatherEdges
{
  NO_GATHER_EDGES, GATHER_IN_EDGES, GATHER_OUT_EDGES, GATHER_ALL_EDGES
};

enum ExpandEdges
{
  NO_EXPAND_EDGES, EXPAND_IN_EDGES, EXPAND_OUT_EDGES, EXPAND_ALL_EDGES
};

enum ApplyVertices
{
  NO_APPLY_VERTICES, APPLY_ALL, APPLY_FRONTIER
};

enum PostApplyVertices
{
  NO_POST_APPLY_VERTICES, POST_APPLY_ALL, POST_APPLY_FRONTIER
};

namespace GASengine
{

  /**
   * Enumeration of global frontier queue configurations
   */
  enum FrontierType
  {
    VERTEX_FRONTIERS,		// O(n) ping-pong global vertex frontiers
    EDGE_FRONTIERS,			// O(m) ping-pong global edge frontiers
    MIXED_FRONTIERS,			// O(n) global vertex frontier, O(m) global edge frontier
    MULTI_GPU_FRONTIERS,			// O(MULTI_GPU_VERTEX_FRONTIER_SCALE * n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier

    MULTI_GPU_VERTEX_FRONTIER_SCALE = 2,
  };

  /**
   * CSR storage management structure for BFS problems.
   */
  template<typename Program, typename _VertexId, typename _SizeT,
      typename _EValue, bool MARK_PREDECESSORS, // Whether to mark predecessors (vs. mark distance from source)
      bool WITH_VALUE>
// Whether to include edge/ndoe value computation with BFS
  struct CsrProblem
  {
    //---------------------------------------------------------------------
    // Typedefs and constants
    //---------------------------------------------------------------------

    typedef ProblemType<Program, // vertex type
        _VertexId,				// VertexId
        _SizeT,					// SizeT
        _EValue,				// Edge Value
        unsigned char,			// VisitedMask
        unsigned char, 			// ValidFlag
        MARK_PREDECESSORS,		// MARK_PREDECESSORS
        WITH_VALUE>             // WITH_VALUE
    ProblemType;

    typedef typename Program::VertexType VertexType;
    typedef typename Program::EdgeType EdgeType;
    typedef typename Program::MiscType MiscType;
    typedef typename Program::GatherType GatherType;
    typedef typename Program::VertexId VertexId;
    typedef typename Program::SizeT SizeT;
    typedef typename ProblemType::VisitedMask VisitedMask;
    typedef typename ProblemType::ValidFlag ValidFlag;
    typedef typename ProblemType::Program::DataType EValue;

    //---------------------------------------------------------------------
    // Helper structures
    //---------------------------------------------------------------------

    /**
     * Graph slice per GPU
     */
    struct GraphSlice
    {
      // GPU index
      int gpu;
      int MPI_rank;
      int pi;
      int pj;

      // Standard CSR device storage arrays
      VertexId *d_column_indices;
      SizeT *d_row_offsets;
      VertexId *d_row_indices;
      SizeT *d_column_offsets;
      VertexId *d_edgeCSC_indices;

//          VertexId *d_row_indices;
//          SizeT *d_column_offsets;

      VertexId *d_labels;				// Source distance
      VertexId *d_preds;               // Predecessor values
      EValue *d_edge_values; // Weight attached to each edge, size equals to the size of d_column_indices
      int num_src;
      int *srcs;
      int outer_iter_num;
      int directed;
      int* d_edgeCountScan;
      int* d_active_flags;
      char* d_changed;
      char* d_bitmap_edgefrontier;
      char* d_bitmap_out;
      char* d_bitmap_prefix;
      char* d_bitmap_assigned;
      char* d_bitmap_in;
      char* d_bitmap_visited;

      GatherType *m_gatherMapTmp;
      GatherType *m_gatherTmp;
      GatherType *m_gatherTmp1;
      GatherType *m_gatherTmp2;
      VertexId *m_gatherDstsTmp;

      VertexType vertex_list;
      EdgeType edge_list;
      // Best-effort mask for keeping track of which vertices we've seen so far
      VisitedMask *d_visited_mask;
      char *d_visit_flags; // Track if same vertex is being expanded inside the same frontier-queue

      // Frontier queues.  Keys track work, values optionally track predecessors.  Only
      // multi-gpu uses triple buffers (single-GPU only uses ping-pong buffers).
      util::TripleBuffer<VertexId, MiscType> frontier_queues;
      SizeT frontier_elements[3];
      SizeT predecessor_elements[3];

      // Flags for filtering duplicates from the edge-frontier queue when partitioning during multi-GPU BFS.
      ValidFlag *d_filter_mask;

      // Number of nodes and edges in slice
      VertexId nodes;
      SizeT edges;

      // CUDA stream to use for processing this slice
      cudaStream_t stream;

      /**
       * Constructor
       */
      GraphSlice(int gpu, int pi, int pj, int directed, cudaStream_t stream) :
          gpu(gpu), pi(pi), pj(pj), directed(directed), d_column_indices(
              NULL), d_row_offsets(NULL), d_edge_values(NULL), d_preds(NULL), d_visited_mask(
              NULL), d_filter_mask(NULL), d_visit_flags(NULL), d_changed(NULL),
              d_bitmap_edgefrontier(NULL), d_bitmap_out(NULL), d_bitmap_prefix(NULL), d_bitmap_assigned(NULL), d_bitmap_in(NULL), d_bitmap_visited(NULL),
              nodes(0), edges(0), stream(stream)
      {
        // Initialize triple-buffer frontier queue lengths
        for (int i = 0; i < 3; i++)
        {
          frontier_elements[i] = 0;
          predecessor_elements[i] = 0;
        }
      }

      /**
       * Destructor
       */
      virtual ~GraphSlice()
      {
        // Set device
        util::B40CPerror(cudaSetDevice(gpu),
            "GpuSlice cudaSetDevice failed", __FILE__, __LINE__);

        cudaFree (m_gatherMapTmp);
        cudaFree(m_gatherTmp);
        if ((Program::gatherOverEdges() == GATHER_ALL_EDGES || directed == 0) && Program::gatherOverEdges() == NO_GATHER_EDGES)
        {
          cudaFree(m_gatherTmp1);
          cudaFree(m_gatherTmp2);
        }

        cudaFree(m_gatherDstsTmp);

        if (d_column_indices)
          util::B40CPerror(cudaFree(d_column_indices),
              "GpuSlice cudaFree d_column_indices failed", __FILE__,
              __LINE__);
        if (d_row_offsets)
          util::B40CPerror(cudaFree(d_row_offsets),
              "GpuSlice cudaFree d_row_offsets failed", __FILE__,
              __LINE__);
//        if (directed == 1)
        if (d_row_indices)
          util::B40CPerror(cudaFree(d_row_indices),
              "GpuSlice cudaFree d_row_indices failed", __FILE__,
              __LINE__);
//        if (directed == 1)
        if (d_column_offsets)
          util::B40CPerror(cudaFree(d_column_offsets),
              "GpuSlice cudaFree d_column_offsets failed",
              __FILE__, __LINE__);
        if (d_edgeCountScan)
          util::B40CPerror(cudaFree(d_edgeCountScan),
              "GpuSlice cudaFree d_edgeCountScan", __FILE__,
              __LINE__);
        if (d_edge_values)
          util::B40CPerror(cudaFree(d_edge_values),
              "GpuSlice cudaFree d_edge_values", __FILE__, __LINE__);

        if (d_active_flags)
          util::B40CPerror(cudaFree(d_active_flags),
              "GpuSlice cudaFree d_active_flags", __FILE__, __LINE__);

        if (d_changed)
          util::B40CPerror(cudaFree(d_changed),
              "GpuSlice cudaFree d_changed", __FILE__, __LINE__);

        if (d_bitmap_edgefrontier)
          util::B40CPerror(cudaFree(d_bitmap_edgefrontier),
              "GpuSlice cudaFree bitmap_edgefrontier", __FILE__, __LINE__);

        if (d_bitmap_out)
          util::B40CPerror(cudaFree(d_bitmap_out),
              "GpuSlice cudaFree d_bitmap_out", __FILE__, __LINE__);

        if (d_bitmap_prefix)
          util::B40CPerror(cudaFree(d_bitmap_prefix),
              "GpuSlice cudaFree d_bitmap_prefix", __FILE__, __LINE__);

        if (d_bitmap_assigned)
          util::B40CPerror(cudaFree(d_bitmap_assigned),
              "GpuSlice cudaFree d_bitmap_assigned", __FILE__, __LINE__);

        if (d_bitmap_in)
          util::B40CPerror(cudaFree(d_bitmap_in),
              "GpuSlice cudaFree d_bitmap_in", __FILE__, __LINE__);

        if (d_bitmap_visited)
          util::B40CPerror(cudaFree(d_bitmap_visited),
              "GpuSlice cudaFree d_bitmap_visited", __FILE__, __LINE__);

        if (d_visited_mask)
          util::B40CPerror(cudaFree(d_visited_mask),
              "GpuSlice cudaFree d_visited_mask failed", __FILE__,
              __LINE__);
        if (d_filter_mask)
          util::B40CPerror(cudaFree(d_filter_mask),
              "GpuSlice cudaFree d_filter_mask failed", __FILE__,
              __LINE__);
        if (d_visit_flags)
          util::B40CPerror(cudaFree(d_visit_flags),
              "GpuSlice cudaFree d_visit_flags failed", __FILE__,
              __LINE__);
        for (int i = 0; i < 3; i++)
        {
          if (frontier_queues.d_keys[i])
            util::B40CPerror(cudaFree(frontier_queues.d_keys[i]),
                "GpuSlice cudaFree frontier_queues.d_keys failed",
                __FILE__, __LINE__);
          if (frontier_queues.d_values[i])
            util::B40CPerror(cudaFree(frontier_queues.d_values[i]),
                "GpuSlice cudaFree frontier_queues.d_values failed",
                __FILE__, __LINE__);
        }

        // Destroy stream
        if (stream)
        {
          util::B40CPerror(cudaStreamDestroy(stream),
              "GpuSlice cudaStreamDestroy failed", __FILE__,
              __LINE__);
        }
      }
    };

    //---------------------------------------------------------------------
    // Members
    //---------------------------------------------------------------------

    // Number of GPUS to be sliced over
    int num_gpus;

    // Size of the graph
    SizeT nodes;
    SizeT edges;
    Config cfg;

    // Set of graph slices (one for each GPU)
    std::vector<GraphSlice*> graph_slices;

    //---------------------------------------------------------------------
    // Methods
    //---------------------------------------------------------------------

    /**
     * Constructor
     */
    CsrProblem(Config cfg) :
        num_gpus(0), nodes(0), edges(0), cfg(cfg)
    {
    }

    /**
     * Destructor
     */
    virtual ~CsrProblem()
    {
      // Cleanup graph slices on the heap
      for (typename std::vector<GraphSlice*>::iterator itr =
          graph_slices.begin(); itr != graph_slices.end(); itr++)
      {
        if (*itr)
          delete (*itr);
      }
    }

    /**
     * Returns index of the gpu that owns the neighbor list of
     * the specified vertex
     */
    template<typename VertexId>
    int GpuIndex(VertexId vertex)
    {
      if (graph_slices.size() == 1)
      {

        // Special case for only one GPU, which may be set as with
        // an ordinal other than 0.
        return graph_slices[0]->gpu;

      }
      else
      {

        return vertex % num_gpus;
      }
    }

    /**
     * Returns the row within a gpu's GraphSlice row_offsets vector
     * for the specified vertex
     */
    template<typename VertexId>
    VertexId GraphSliceRow(VertexId vertex)
    {
      return vertex / num_gpus;
    }

    /**
     * Extract into a single host vector the BFS results disseminated across
     * all GPUs
     */
    cudaError_t ExtractResults(EValue *h_values)
    {
      cudaError_t retval = cudaSuccess;

      do
      {
        if (graph_slices.size() == 1)
        {
          // Set device
          if (util::B40CPerror(cudaSetDevice(graph_slices[0]->gpu),
              "CsrProblem cudaSetDevice failed", __FILE__, __LINE__))
            break;

          Program::extractResult(graph_slices[0]->vertex_list, h_values);

        }
        else
        {
          printf("Multi GPU is not supported yet!\n");
          exit(0);
        }
      }
      while (0);

      return retval;
    }

    /**
     * Initialize from host CSR problem
     */
    cudaError_t FromHostProblem(
        bool stream_from_host,		// Only meaningful for single-GPU BFS
        SizeT nodes,
        SizeT edges,
        VertexId *h_column_indices,
        SizeT *h_row_offsets,
        EValue *h_edge_values,
        VertexId *h_row_indices,
        SizeT *h_column_offsets,
        int num_gpus,
        int directed,
        int device_id,
        int rank_id)
    {
      int device = device_id; //cfg.getParameter<int>("device");
      cudaError_t retval = cudaSuccess;
      this->nodes = nodes;
      this->edges = edges;

      this->num_gpus = num_gpus;

      do
      {
        if (num_gpus <= 1)
        {

          // Create a single GPU slice for the currently-set gpu
          int gpu = device;
//              if (retval = util::B40CPerror(cudaGetDevice(&gpu), "CsrProblem cudaGetDevice failed", __FILE__, __LINE__)) break;
          if (retval = util::B40CPerror(cudaSetDevice(gpu),
              "CsrProblem cudaGetDevice failed", __FILE__, __LINE__))
            break;
//              printf("Running on device %d\n", device);
          graph_slices.push_back(new GraphSlice(gpu, 0, 0, directed, 0));
          graph_slices[0]->nodes = nodes;
          graph_slices[0]->edges = edges;

          cudaMalloc((void**) &graph_slices[0]->m_gatherMapTmp, (graph_slices[0]->edges + graph_slices[0]->nodes) * sizeof(GatherType));
          cudaMalloc((void**) &graph_slices[0]->m_gatherTmp, graph_slices[0]->nodes * sizeof(GatherType));
          if ((Program::gatherOverEdges() == GATHER_ALL_EDGES || directed == 0) && Program::gatherOverEdges() != NO_GATHER_EDGES)
          {
            cudaMalloc((void**) &graph_slices[0]->m_gatherTmp1, graph_slices[0]->nodes * sizeof(GatherType));
            //        cudaMemset(m_gatherTmp1, 0, graph_slice->nodes * sizeof(GatherType) );
            cudaMalloc((void**) &graph_slices[0]->m_gatherTmp2, graph_slices[0]->nodes * sizeof(GatherType));
          }
          cudaMalloc((void**) &graph_slices[0]->m_gatherDstsTmp, (graph_slices[0]->edges + graph_slices[0]->nodes) * sizeof(VertexId));

          //      thrust::device_vector<int> d_vertex_ids = thrust::device_vector<int>(graph_slice->nodes);
          //      thrust::sequence(d_vertex_ids.begin(), d_vertex_ids.end());
          int memset_block_size = 256;
          int memset_grid_size_max = 32 * 1024;              // 32K CTAs
          int memset_grid_size;

          memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[0]->nodes + memset_block_size - 1) / memset_block_size);

          //init m_gatherTmp, necessary for CC!!!
          util::MemsetKernel<GatherType><<<memset_grid_size,
          memset_block_size, 0, graph_slices[0]->stream>>>(
              graph_slices[0]->m_gatherTmp, Program::INIT_VALUE,
              graph_slices[0]->nodes);

          //
          //Device mem allocations
          //
          printf("GPU %d column_indices: %lld elements (%lld bytes)\n",
              graph_slices[0]->gpu,
              (unsigned long long) (graph_slices[0]->edges),
              (unsigned long long) (graph_slices[0]->edges
                  * sizeof(VertexId) * sizeof(SizeT)));

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_column_indices,
                  graph_slices[0]->edges * sizeof(VertexId)),
              "CsrProblem cudaMalloc d_column_indices failed",
              __FILE__, __LINE__))
            break;

          printf("GPU %d row_offsets: %lld elements (%lld bytes)\n",
              graph_slices[0]->gpu,
              (unsigned long long) (graph_slices[0]->nodes + 1),
              (unsigned long long) (graph_slices[0]->nodes + 1)
                  * sizeof(SizeT));

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_row_offsets,
                  (graph_slices[0]->nodes + 1) * sizeof(SizeT)),
              "CsrProblem cudaMalloc d_row_offsets failed", __FILE__,
              __LINE__))
            break;

//          if (directed)
          {
            if (retval = util::B40CPerror(
                cudaMalloc((void**) &graph_slices[0]->d_row_indices,
                    graph_slices[0]->edges * sizeof(VertexId)),
                "CsrProblem cudaMalloc d_row_indices failed",
                __FILE__, __LINE__))
              break;

            if (retval = util::B40CPerror(
                cudaMalloc(
                    (void**) &graph_slices[0]->d_column_offsets,
                    (graph_slices[0]->nodes + 1)
                        * sizeof(SizeT)),
                "CsrProblem cudaMalloc d_column_offsets failed",
                __FILE__, __LINE__))
              break;

            if (retval = util::B40CPerror(
                cudaMalloc(
                    (void**) &graph_slices[0]->d_edgeCSC_indices,
                    graph_slices[0]->edges * sizeof(VertexId)),
                "CsrProblem cudaMalloc d_edgeCSC_indices failed",
                __FILE__, __LINE__))
              break;
          }

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_edgeCountScan,
                  (graph_slices[0]->nodes + 1) * sizeof(SizeT)),
              "CsrProblem cudaMalloc d_edgeCountScan failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_active_flags,
                  (graph_slices[0]->nodes) * sizeof(int)),
              "CsrProblem cudaMalloc d_active_flags failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_changed,
                  (graph_slices[0]->nodes) * sizeof(char)),
              "CsrProblem cudaMalloc d_changed failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_visit_flags,
                  (graph_slices[0]->nodes) * sizeof(char)),
              "CsrProblem cudaMalloc d_visit_flags failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_edgefrontier,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_edgefrontier failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_out,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_out failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_prefix,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_prefix failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_assigned,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_assigned failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_in,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_in failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_bitmap_visited,
                  (graph_slices[0]->nodes + 8 - 1) / 8),
              "CsrProblem cudaMalloc d_bitmap_visited failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[0]->d_edge_values,
                  graph_slices[0]->edges * sizeof(VertexId)),
              "CsrProblem cudaMalloc d_edge_values failed", __FILE__,
              __LINE__))
            break;

          //
          //Initializations
          //

          double starttransfer = omp_get_wtime();
          if (retval = util::B40CPerror(
              cudaMemcpy(graph_slices[0]->d_column_indices,
                  h_column_indices,
                  graph_slices[0]->edges * sizeof(VertexId),
                  cudaMemcpyHostToDevice),
              "CsrProblem cudaMemcpy d_column_indices failed",
              __FILE__, __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMemcpy(graph_slices[0]->d_row_offsets,
                  h_row_offsets,
                  (graph_slices[0]->nodes + 1) * sizeof(SizeT),
                  cudaMemcpyHostToDevice),
              "CsrProblem cudaMemcpy d_row_offsets failed", __FILE__,
              __LINE__))
            break;

          if (retval = util::B40CPerror(
              cudaMemcpy(graph_slices[0]->d_edge_values,
                  h_edge_values,
                  graph_slices[0]->edges * sizeof(VertexId),
                  cudaMemcpyHostToDevice),
              "CsrProblem cudaMemcpy d_edge_values failed",
              __FILE__, __LINE__))
            break;

          cudaDeviceSynchronize();
          double endtransfer = omp_get_wtime();
          printf("CPU to GPU memory transfer time: %f ms\n", (endtransfer - starttransfer) * 1000.0);

//          if (directed)
          {
            thrust::device_ptr<SizeT> d_row_offsets_ptr = thrust::device_pointer_cast(graph_slices[0]->d_row_offsets);
            thrust::device_ptr<VertexId> d_row_indices_ptr = thrust::device_pointer_cast(graph_slices[0]->d_row_indices);
            thrust::device_ptr<SizeT> d_column_offsets_ptr = thrust::device_pointer_cast(graph_slices[0]->d_column_offsets);
            thrust::device_ptr<VertexId> d_column_indices_ptr = thrust::device_pointer_cast(graph_slices[0]->d_column_indices);
            thrust::device_ptr<EValue> d_edge_values_ptr = thrust::device_pointer_cast(graph_slices[0]->d_edge_values);
            thrust::device_ptr<VertexId> d_edgeCSC_indices_ptr = thrust::device_pointer_cast(graph_slices[0]->d_edgeCSC_indices);

            offsets_to_indices(graph_slices[0]->nodes + 1, graph_slices[0]->edges, d_row_offsets_ptr, d_row_indices_ptr);
            sort_by_column(graph_slices[0]->edges, graph_slices[0]->nodes + 1, d_column_indices_ptr, d_row_indices_ptr, d_column_offsets_ptr, d_edgeCSC_indices_ptr);

//            printf("edge values:\n");
//            for(int i=0; i<graph_slices[0]->edges; i++)
//            {
//              std::cout << d_edge_values_ptr[i] << " ";
//            }
//            printf("\n");

//            printf("CSC matrix:\n");
//            for (int i = 0; i < graph_slices[0]->nodes; i++)
//            {
//              printf("%d: ", i);
//              for (int j = d_column_offsets_ptr[i]; j < d_column_offsets_ptr[i + 1]; j++)
//                std::cout << d_row_indices_ptr[j] << " ";
//              printf("\n");
//            }
          }

          memset_grid_size = B40C_MIN(memset_grid_size_max, (graph_slices[0]->nodes + memset_block_size - 1) / memset_block_size);
          util::MemsetKernel<int><<<memset_grid_size,
          memset_block_size, 0, graph_slices[0]->stream>>>(
              graph_slices[0]->d_active_flags, 0,
              graph_slices[0]->nodes);

          if (retval = util::B40CPerror(cudaThreadSynchronize(),
              "MemsetKernel d_active_flags failed", __FILE__, __LINE__))
            return retval;

          util::MemsetKernel<char><<<memset_grid_size,
          memset_block_size, 0, graph_slices[0]->stream>>>(
              graph_slices[0]->d_changed, 0,
              graph_slices[0]->nodes);

          if (retval = util::B40CPerror(cudaThreadSynchronize(),
              "MemsetKernel d_changed failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_edgefrontier, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_edgefrontier failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_out, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_out failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_prefix, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_prefix failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_assigned, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_assigned failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_in, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_out failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_bitmap_visited, 0, (graph_slices[0]->nodes + 8 - 1) / 8),
              "Memset d_bitmap_visited failed", __FILE__, __LINE__))
            return retval;

          if (retval = util::B40CPerror(cudaMemset(graph_slices[0]->d_visit_flags, 0, graph_slices[0]->nodes * sizeof(char)),
              "Memset d_visit_flags failed", __FILE__, __LINE__))
            return retval;

        }
        else //TODO: multiple GPU
        {
        }

      }
      while (0);

      return retval;
    }

    /**
     * Performs any initialization work needed for this problem type.  Must be called
     * prior to each search
     */
    cudaError_t Reset(FrontierType frontier_type, double queue_sizing)
    {
      cudaError_t retval = cudaSuccess;
//      printf("Starting vertex: %d\n", src);

      for (int gpu = 0; gpu < num_gpus; gpu++)
      {

        // Set device
        if (retval = util::B40CPerror(cudaSetDevice(graph_slices[gpu]->gpu),
            "CsrProblem cudaSetDevice failed", __FILE__, __LINE__))
          return retval;

        //
        // Allocate visited masks for the entire graph if necessary
        //

        int visited_mask_bytes = ((nodes * sizeof(VisitedMask)) + 8 - 1)
            / 8;				// round up to the nearest VisitedMask
        int visited_mask_elements = visited_mask_bytes
            * sizeof(VisitedMask);
        if (!graph_slices[gpu]->d_visited_mask)
        {

//              printf("GPU %d visited mask: %lld elements (%lld bytes)\n", graph_slices[gpu]->gpu, (unsigned long long) visited_mask_elements, (unsigned long long) visited_mask_bytes);

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[gpu]->d_visited_mask,
                  visited_mask_bytes),
              "CsrProblem cudaMalloc d_visited_mask failed", __FILE__,
              __LINE__))
            return retval;
        }

        //
        // Allocate frontier queues if necessary
        //

        // Determine frontier queue sizes
        SizeT new_frontier_elements[3] = { 0, 0, 0 };
        SizeT new_predecessor_elements[3] = { 0, 0, 0 };

        switch (frontier_type)
        {
          case VERTEX_FRONTIERS:
            // O(n) ping-pong global vertex frontiers
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes)
                * queue_sizing;
            new_frontier_elements[1] = new_frontier_elements[0];
            break;

          case EDGE_FRONTIERS:
            // O(m) ping-pong global edge frontiers
            new_frontier_elements[0] = double(graph_slices[gpu]->edges)
                * queue_sizing;
            new_frontier_elements[1] = new_frontier_elements[0];
            new_frontier_elements[2] = new_frontier_elements[0];
            new_predecessor_elements[0] = new_frontier_elements[0];
            new_predecessor_elements[1] = new_frontier_elements[1];
            new_predecessor_elements[2] = new_frontier_elements[2];
            break;

          case MIXED_FRONTIERS:
            // O(n) global vertex frontier, O(m) global edge frontier
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes)
                * queue_sizing;
            new_frontier_elements[1] = double(graph_slices[gpu]->edges)
                * queue_sizing;
            new_predecessor_elements[1] = new_frontier_elements[1];
            break;

          case MULTI_GPU_FRONTIERS:
            // O(n) global vertex frontier, O(m) global edge frontier, O(m) global sorted, filtered edge frontier
            new_frontier_elements[0] = double(graph_slices[gpu]->nodes)
                * MULTI_GPU_VERTEX_FRONTIER_SCALE * queue_sizing;
            new_frontier_elements[1] = double(graph_slices[gpu]->edges)
                * queue_sizing;
            new_frontier_elements[2] = new_frontier_elements[1];
            new_predecessor_elements[1] = new_frontier_elements[1];
            new_predecessor_elements[2] = new_frontier_elements[2];
            break;
        }

        // Iterate through global frontier queue setups
        for (int i = 0; i < 3; i++)
        {

          // Allocate frontier queue if not big enough
          if (graph_slices[gpu]->frontier_elements[i]
              < new_frontier_elements[i])
          {

            // Free if previously allocated
            if (graph_slices[gpu]->frontier_queues.d_keys[i])
            {
              if (retval =
                  util::B40CPerror(
                      cudaFree(
                          graph_slices[gpu]->frontier_queues.d_keys[i]),
                      "GpuSlice cudaFree frontier_queues.d_keys failed",
                      __FILE__, __LINE__))
                return retval;
            }

            graph_slices[gpu]->frontier_elements[i] =
                new_frontier_elements[i];

            if (retval =
                util::B40CPerror(
                    cudaMalloc(
                        (void**) &graph_slices[gpu]->frontier_queues.d_keys[i],
                        graph_slices[gpu]->frontier_elements[i]
                            * sizeof(VertexId)),
                    "CsrProblem cudaMalloc frontier_queues.d_keys failed",
                    __FILE__, __LINE__))
              return retval;
          }

          // Allocate predecessor queue if not big enough
          if (graph_slices[gpu]->predecessor_elements[i]
              < new_predecessor_elements[i])
          {

            // Free if previously allocated
            if (graph_slices[gpu]->frontier_queues.d_values[i])
            {
              if (retval =
                  util::B40CPerror(
                      cudaFree(
                          graph_slices[gpu]->frontier_queues.d_values[i]),
                      "GpuSlice cudaFree frontier_queues.d_values failed",
                      __FILE__, __LINE__))
                return retval;
            }

            graph_slices[gpu]->predecessor_elements[i] =
                new_predecessor_elements[i];

            if (retval =
                util::B40CPerror(
                    cudaMalloc(
                        (void**) &graph_slices[gpu]->frontier_queues.d_values[i],
                        graph_slices[gpu]->predecessor_elements[i]
                            * sizeof(MiscType)),
                    "CsrProblem cudaMalloc frontier_queues.d_values failed",
                    __FILE__, __LINE__))
              return retval;
          }
        }

        //
        // Allocate duplicate filter mask if necessary (for multi-gpu)
        //

        if ((frontier_type == MULTI_GPU_FRONTIERS)
            && (!graph_slices[gpu]->d_filter_mask))
        {

          if (retval = util::B40CPerror(
              cudaMalloc((void**) &graph_slices[gpu]->d_filter_mask,
                  graph_slices[gpu]->frontier_elements[1]
                      * sizeof(ValidFlag)),
              "CsrProblem cudaMalloc d_filter_mask failed", __FILE__,
              __LINE__))
            return retval;
        }

        //only 1 sourc is allowed now

//        _Program::Initialize(graph_slices[gpu]->nodes,
//            graph_slices[gpu]->edges, 1, graph_slices[gpu]->srcs,
//            graph_slices[gpu]->d_row_offsets,
//            graph_slices[gpu]->d_column_indices,
//            graph_slices[gpu]->d_column_offsets,
//            graph_slices[gpu]->d_row_indices,
//            graph_slices[gpu]->d_edge_values,
//            graph_slices[gpu]->vertex_list,
//            graph_slices[gpu]->edge_list,
//            graph_slices[gpu]->frontier_queues.d_keys,
//            graph_slices[gpu]->frontier_queues.d_values);

        int memset_block_size = 256;
        int memset_grid_size_max = 32 * 1024;	// 32K CTAs
        int memset_grid_size;

        // Initialize d_visited_mask elements to 0
        memset_grid_size =
            B40C_MIN(memset_grid_size_max, (visited_mask_elements + memset_block_size - 1) / memset_block_size);

        util::MemsetKernel<VisitedMask><<<memset_grid_size,
        memset_block_size, 0, graph_slices[gpu]->stream>>>(
            graph_slices[gpu]->d_visited_mask, 0,
            visited_mask_elements);

        if (retval = util::B40CPerror(cudaThreadSynchronize(),
            "MemsetKernel failed", __FILE__, __LINE__))
          return retval;
      }

      return retval;
    }
  };

} // namespace GASengine

