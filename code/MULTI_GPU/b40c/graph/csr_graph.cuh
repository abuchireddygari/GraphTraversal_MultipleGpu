/******************************************************************************
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
 * Thanks!
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
 * Simple CSR sparse graph data structure
 ******************************************************************************/

#pragma once

#include <time.h>
#include <stdio.h>

#include <algorithm>

#include <b40c/util/error_utils.cuh>

namespace b40c
{
  namespace graph
  {

    /**
     * CSR sparse format graph
     */
    template<typename VertexId, typename Value, typename SizeT>
    struct CsrGraph
    {
      SizeT nodes;
      SizeT edges;

      SizeT *row_offsets;
      VertexId *column_indices;
      SizeT *column_offsets;
      VertexId *row_indices;
      Value *edge_values;
      Value *node_values;
      VertexId *from_nodes;
      VertexId *to_nodes;

      /**
       * Constructor
       */
      CsrGraph(bool pinned = false)
      {
        nodes = 0;
        edges = 0;
        row_offsets = NULL;
        column_indices = NULL;
        column_offsets = NULL;
        row_indices = NULL;
        edge_values = NULL;
        node_values = NULL;
      }

      template<bool LOAD_VALUES>
      void FromScratch(SizeT nodes, SizeT edges, int undirected)
      {
        this->nodes = nodes;
        this->edges = edges;

        // Put our graph in regular memory
        row_offsets = (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
        column_indices = (VertexId*) malloc(sizeof(VertexId) * edges);
        if (!undirected)
        {
          column_offsets = (SizeT*) malloc(sizeof(SizeT) * (nodes + 1));
          row_indices = (VertexId*) malloc(sizeof(VertexId) * edges);
        }
        edge_values = (LOAD_VALUES) ? (Value*) malloc(sizeof(Value) * edges) : NULL;
        node_values = (LOAD_VALUES) ? (Value*) malloc(sizeof(Value) * nodes) : NULL;

      }

      /**
       * Build CSR graph from sorted COO graph
       */
      template<bool LOAD_VALUES, typename Tuple>
      void FromCoo(Tuple *coo, SizeT coo_nodes, SizeT coo_edges, int undirected = false, bool ordered_rows = false)
      {
//        printf("  Converting %d vertices, %d directed edges (%s tuples) to CSR format... ", coo_nodes, coo_edges, ordered_rows ? "ordered" : "unordered");
        time_t mark1 = time(NULL);
        fflush (stdout);

        FromScratch<LOAD_VALUES>(coo_nodes, coo_edges, undirected);

        std::stable_sort(coo, coo + coo_edges, DimacsTupleCompare<Tuple>);

//        printf("After sort:\n");
//        for (int i = 0; i < coo_edges; i++)
//        {
//          printf("%d %d %d\n", coo[i].row, coo[i].col, coo[i].val);
//        }

        VertexId prev_row = -1;
        for (SizeT edge = 0; edge < edges; edge++)
        {

          VertexId current_row = coo[edge].row;

          // Fill in rows up to and including the current row
          for (VertexId row = prev_row + 1; row <= current_row; row++)
          {
            row_offsets[row] = edge;
          }
          prev_row = current_row;

          column_indices[edge] = coo[edge].col;
          if (LOAD_VALUES)
          {
            edge_values[edge] = coo[edge].val;
//            coo[edge].Val(edge_values[edge]);
          }
          else
            edge_values[edge] = 1;
        }

        // Fill out any trailing edgeless nodes (and the end-of-list element)
        for (VertexId row = prev_row + 1; row <= nodes; row++)
        {
          row_offsets[row] = edges;
        }

//        printf("After CSR:\n");
//        for (int i = 0; i < coo_edges; i++)
//        {
//          printf("%d\n", edge_values[i]);
//        }

        if (!undirected)
        {
          // Sort COO by col
          std::stable_sort(coo, coo + coo_edges, DimacsTupleCompare2<Tuple>);

          VertexId prev_col = -1;
          for (SizeT edge = 0; edge < edges; edge++)
          {

            VertexId current_col = coo[edge].col;

            // Fill in rows up to and including the current row
            for (VertexId col = prev_col + 1; col <= current_col; col++)
            {
              column_offsets[col] = edge;
            }
            prev_col = current_col;

            row_indices[edge] = coo[edge].row;
//            if (LOAD_VALUES)
//            {
//              edge_values[edge] = coo[edge].val;
////            coo[edge].Val(edge_values[edge]);
//            }
//            else
//              edge_values[edge] = 1;
          }

          // Fill out any trailing edgeless nodes (and the end-of-list element)
          for (VertexId col = prev_col + 1; col <= nodes; col++)
          {
            column_offsets[col] = edges;
          }
        }

        time_t mark2 = time(NULL);
//        printf("Done converting (%ds).\n", (int) (mark2 - mark1));
        fflush(stdout);
      }

      /**
       * Print log-histogram
       */
      void PrintHistogram()
      {
        fflush (stdout);

        // Initialize
        int log_counts[32];
        for (int i = 0; i < 32; i++)
        {
          log_counts[i] = 0;
        }

        // Scan
        int max_log_length = -1;
        for (VertexId i = 0; i < nodes; i++)
        {

          SizeT length = row_offsets[i + 1] - row_offsets[i];

          int log_length = -1;
          while (length > 0)
          {
            length >>= 1;
            log_length++;
          }
          if (log_length > max_log_length)
          {
            max_log_length = log_length;
          }

          log_counts[log_length + 1]++;
        }
        printf("\nDegree Histogram (%lld vertices, %lld directed edges):\n", (long long) nodes, (long long) edges);
        for (int i = -1; i < max_log_length + 1; i++)
        {
          printf("\tDegree 2^%i: %d (%.2f%%)\n", i, log_counts[i + 1], (float) log_counts[i + 1] * 100.0 / nodes);
        }
        printf("\n");
        fflush(stdout);
      }

      /**
       * Display CSR graph to console
       */
      void DisplayGraph()
      {
        printf("Input Graph:\n");
        for (VertexId node = 0; node < nodes; node++)
        {
          printf("%d", node);
          printf(": ");
          for (SizeT edge = row_offsets[node]; edge < row_offsets[node + 1]; edge++)
          {
//            PrintValue(column_indices[edge]);
            printf("%d", column_indices[edge]);
            printf(", ");
          }
          printf("\n");
        }

        printf("Input Graph CSC:\n");
        for (VertexId node = 0; node < nodes; node++)
        {

//          PrintValue(node);
          printf("%d", node);
          printf(": ");
          for (SizeT edge = column_offsets[node]; edge < column_offsets[node + 1]; edge++)
          {
//            PrintValue(row_indices[edge]);
            printf("%d", row_indices[edge]);
            printf(", ");
          }
          printf("\n");
        }
      }

      /**
       * Deallocates graph
       */
      void Free()
      {
        if (row_offsets)
        {
          free(row_offsets);
          row_offsets = NULL;
        }
        if (column_indices)
        {

          free(column_indices);

          column_indices = NULL;
        }

        if (column_offsets)
        {

          free(column_offsets);

          column_offsets = NULL;
        }
        if (row_indices)
        {

          free(row_indices);

          row_indices = NULL;
        }

        if (from_nodes)
        {

          free(from_nodes);

          from_nodes = NULL;
        }

        if (to_nodes)
        {

          free(to_nodes);

          to_nodes = NULL;
        }

        if (edge_values)
        {

          free(edge_values);

          edge_values = NULL;
        }
        if (node_values)
        {

          free(node_values);

          node_values = NULL;
        }

        nodes = 0;
        edges = 0;
      }

      /**
       * Destructor
       */
      ~CsrGraph()
      {
//        Free();
      }
    };

  } // namespace graph
} // namespace b40c
