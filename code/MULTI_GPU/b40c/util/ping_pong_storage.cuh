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
 * Thanks!
 * 
 ******************************************************************************/


/******************************************************************************
 *  Storage wrapper for double-buffered vectors (deprecated).
 ******************************************************************************/

#pragma once

#include <b40c/util/multiple_buffering.cuh>

namespace b40c {
namespace util {

/**
 * Ping-pong buffer (a.k.a. page-flip, double-buffer, etc.).
 * Deprecated: see b40c::util::DoubleBuffer instead.
 */
template <
	typename KeyType,
	typename ValueType = util::NullType>
struct PingPongStorage : DoubleBuffer<KeyType, ValueType>
{
	typedef DoubleBuffer<KeyType, ValueType> ParentType;

	// Constructor
	PingPongStorage() : ParentType() {}

	// Constructor
	PingPongStorage(
		KeyType* keys) : ParentType(keys) {}

	// Constructor
	PingPongStorage(
		KeyType* keys,
		ValueType* values) : ParentType(keys, values) {}

	// Constructor
	PingPongStorage(
		KeyType* keys0,
		KeyType* keys1,
		ValueType* values0,
		ValueType* values1) : ParentType(keys0, keys1, values0, values1) {}
};


} // namespace util
} // namespace b40c

