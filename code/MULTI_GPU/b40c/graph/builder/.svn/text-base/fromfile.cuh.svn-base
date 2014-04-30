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

/******************************************************************************
 * Graph Construction from File:
 * firstline: node_num, edge_num
 * restlines: from_vertex, to_vertex, value
 ******************************************************************************/

#pragma once

#include <math.h>
#include <time.h>
#include <stdio.h>
#include <string.h>

#include <b40c/graph/builder/utils.cuh>

namespace b40c {
namespace graph {
namespace builder {


/**
 * Builds a CSR graph by reading data from files in ~/dataset    
 * Returns 0 on success, 1 on failure.
 */
template<bool LOAD_VALUES, typename VertexId, typename Value, typename SizeT>
int BuildGraphFromFile(
	char* file_name,
	CsrGraph<VertexId, Value, SizeT> &csr_graph)
{ 
	typedef CooEdgeTuple<VertexId, Value> EdgeTupleType;
    VertexId nodes;
    VertexId edges;

	printf("  Reading Data from %s and converting into COO format... ", 
		file_name);
	fflush(stdout);

	// Construct COO graph
    FILE* ifp;
    ifp = fopen(file_name, "r");
    if (ifp == NULL)
    {
        fprintf(stderr, "Cannot open input file.\n");
        exit(1);
    }

    if (fscanf(ifp, "%d", &nodes) < 0) exit(-1);
    if (fscanf(ifp, "%d", &edges) < 0) exit(-1);
    int directed_edges = edges;
    long long int from, to;
    float value;
	EdgeTupleType *coo = (EdgeTupleType*) malloc(sizeof(EdgeTupleType) * directed_edges);
    for ( int i = 0; i < directed_edges; ++i )
    {
        if (fscanf(ifp, "%Ld %Ld %f", &from, &to, &value) < 0) exit(-1);
		coo[i].row = from;
		coo[i].col = to;
		if (LOAD_VALUES)
		    coo[i].val = value;
	}
    fclose(ifp);
	fflush(stdout);

	// Convert sorted COO to CSR
	csr_graph.template FromCoo<LOAD_VALUES>(coo, nodes, directed_edges);
	free(coo);

	return 0;
}


} // namespace builder
} // namespace graph
} // namespace b40c
