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

#ifndef PARTITIONER_H
#define	PARTITIONER_H

#include <b40c/graph/csr_graph.cuh>
#include <vector>
using namespace b40c;
using namespace graph;
using namespace std;

template<typename Program>
class Partitioner
{
  typedef typename Program::VertexId VertexId;
  typedef typename Program::DataType DataType;
  typedef typename Program::SizeT SizeT;
  typedef CooEdgeTuple<VertexId, DataType> EdgeTupleType;
  CsrGraph<VertexId, DataType, SizeT>* csr_graph;
  int num_parts;

public:
  //constructor

  Partitioner(CsrGraph<VertexId, DataType, SizeT>* csr_graph, int num_parts) :
      csr_graph(csr_graph), num_parts(num_parts)
  {
  }

  //destructor

  ~Partitioner()
  {
  }

  void partition(vector<EdgeTupleType*> &coos, vector<long long>& part_count)
  {

    SizeT nodes = csr_graph->nodes;
    SizeT num_part_1d = sqrt(num_parts);
    SizeT num_vert_per_part_1d = (nodes + num_part_1d - 1) / num_part_1d;

    printf("nodes=%d, num_part_1d=%d, num_vert_per_part_1d=%d\n", nodes, num_part_1d, num_vert_per_part_1d);

    coos.clear();
    coos.resize(num_parts, NULL);
    part_count.clear();
    part_count.resize(num_parts, 0);

    for (SizeT vid = 0; vid < nodes; vid++)
    {
      SizeT start = csr_graph->row_offsets[vid];
      SizeT end = csr_graph->row_offsets[vid + 1];
      for (SizeT eid = start; eid < end; eid++)
      {
        VertexId src = vid;
        VertexId dst = csr_graph->column_indices[eid];
        SizeT pj = src / num_vert_per_part_1d;
        SizeT pi = dst / num_vert_per_part_1d;
        SizeT part = pi * num_part_1d + pj;
        part_count[part]++;
      }
    }

    for (SizeT i = 0; i < num_parts; i++)
    {
      coos[i] = new EdgeTupleType[part_count[i]];
      part_count[i] = 0;
    }

    for (SizeT vid = 0; vid < nodes; vid++)
    {
      SizeT start = csr_graph->row_offsets[vid];
      SizeT end = csr_graph->row_offsets[vid + 1];

      for (SizeT eid = start; eid < end; eid++)
      {
        VertexId src = vid;
        VertexId dst = csr_graph->column_indices[eid];
        DataType value = csr_graph->edge_values[eid];
        SizeT pj = src / num_vert_per_part_1d;
        SizeT pi = dst / num_vert_per_part_1d;
        SizeT part = pi * num_part_1d + pj;
        coos[part][part_count[part]] = EdgeTupleType(src - num_vert_per_part_1d*pj, dst - num_vert_per_part_1d*pi, value);
        part_count[part]++;
      }
    }

//    for(int i=0; i<coos.size(); i++)
//    {
//      printf("Partition %d: ", i);
//      for(int j=0; j<part_count[i]; j++)
//      {
//        printf("(%d, %d), ", coos[i][j].row, coos[i][j].col);
//      }
//      printf("\n");
//    }
  }
};

#endif	/* PARTITIONER_H */

