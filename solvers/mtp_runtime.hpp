// SPDX-License-Identifier: MIT
// Runtime contract shared by all ten campaign solvers (refactored + evolved).
//
//   ./solver SEED < instance.txt > solution.txt
//
// * SEED is the first command-line argument (unsigned integer). Seeded solvers
//   derive every RNG state from it; deterministic solvers accept and ignore it.
// * Wall-clock budget: 600 s for the whole process (read + search + write).
//   The timer starts in the first statement of main(); search stops at
//   SEARCH_LIMIT_SECONDS = 590 s, leaving 10 s to write the best solution.
// * Single-threaded: nothing in this header (or the solvers) spawns threads.
// * Skill capacity: skill ids index fixed-size arrays of size NS. The guard in
//   skill_id() stops the program with a non-zero exit code *before* an
//   out-of-range id can be produced.

#ifndef MTP_RUNTIME_HPP
#define MTP_RUNTIME_HPP

#include <chrono>
#include <cstdlib>
#include <iostream>
#include <string>
#include <unordered_map>

namespace mtp {

// Largest number of distinct skill names (contributor skills and project-role
// skills share one index space) over the 611 campaign instances is 919 (type2_009;
// measured with campaign/instances.py). NS adds no headroom on purpose: the
// guard below reports any instance that needs more.
constexpr int NS = 919;

// 590 s in every campaign build. The macro exists only so tests can build a
// short-deadline variant (-DMTP_SEARCH_LIMIT_SECONDS=5); build_solvers.sh never sets it.
#ifndef MTP_SEARCH_LIMIT_SECONDS
#define MTP_SEARCH_LIMIT_SECONDS 590.0
#endif
constexpr double SEARCH_LIMIT_SECONDS = MTP_SEARCH_LIMIT_SECONDS;

constexpr int EXIT_BAD_ARGS = 2;
constexpr int EXIT_SKILL_CAPACITY = 3;

using Clock = std::chrono::steady_clock;

inline Clock::time_point &start_time() {
	static Clock::time_point t0;
	return t0;
}

// Call as the first statement of main().
inline void start_timer() { start_time() = Clock::now(); }

inline double elapsed_seconds() {
	return std::chrono::duration<double>(Clock::now() - start_time()).count();
}

// True once the search budget is spent. Cheap (one steady_clock read), so it
// is safe to call inside inner loops.
inline bool time_up() { return elapsed_seconds() >= SEARCH_LIMIT_SECONDS; }

// Parses argv[1] as the seed. Seeded solvers pass required=true and fail with
// EXIT_BAD_ARGS if it is missing; deterministic solvers pass required=false.
inline unsigned parse_seed(int argc, char **argv, bool required) {
	if(argc < 2) {
		if(!required) return 0u;
		std::cerr << "ERROR: missing seed. Usage: " << (argc ? argv[0] : "solver")
		          << " SEED < instance.txt > solution.txt" << std::endl;
		std::exit(EXIT_BAD_ARGS);
	}
	const std::string arg(argv[1]);
	char *end = nullptr;
	const unsigned long v = std::strtoul(arg.c_str(), &end, 10);
	if(arg.empty() || arg[0] == '-' || *end != '\0' || v > 0xFFFFFFFFul) {
		std::cerr << "ERROR: seed must be an unsigned 32-bit integer, got '" << arg << "'" << std::endl;
		std::exit(EXIT_BAD_ARGS);
	}
	return static_cast<unsigned>(v);
}

// Returns the id of skill `name`, assigning the next free id on first sight.
// Exits with EXIT_SKILL_CAPACITY before an id >= NS could ever be created.
inline int skill_id(std::unordered_map<std::string, int> &s2i, const std::string &name) {
	const auto it = s2i.find(name);
	if(it != s2i.end()) return it->second;
	const int id = static_cast<int>(s2i.size());
	if(id >= NS) {
		std::cerr << "ERROR: instance has more than NS=" << NS
		          << " distinct skills (first overflowing skill: '" << name
		          << "'); increase mtp::NS in solvers/mtp_runtime.hpp and recompile." << std::endl;
		std::exit(EXIT_SKILL_CAPACITY);
	}
	s2i.emplace(name, id);
	return id;
}

}  // namespace mtp

#endif
