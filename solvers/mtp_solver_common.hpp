// Copyright (C) 2026
// SPDX-License-Identifier: MIT
// Shared experiment-interface helpers for the MTP solver campaign.

#ifndef MTP_SOLVER_COMMON_HPP
#define MTP_SOLVER_COMMON_HPP

#include <algorithm>
#include <chrono>
#include <cstdint>
#include <cstdlib>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

struct SolverOptions {
	unsigned seed = 42;
	double time_limit_seconds = std::numeric_limits<double>::infinity();
	int threads = 1;
};

inline std::string require_arg_value(int argc, char **argv, int &i, const std::string &name) {
	std::string arg(argv[i]);
	const std::string prefix = name + "=";
	if(arg.rfind(prefix, 0) == 0) return arg.substr(prefix.size());
	if(i + 1 >= argc) throw std::invalid_argument("Missing value for " + name);
	return std::string(argv[++i]);
}

inline SolverOptions parse_solver_options(int argc, char **argv) {
	SolverOptions options;
	for(int i = 1; i < argc; ++i) {
		std::string arg(argv[i]);
		if(arg == "--seed" || arg.rfind("--seed=", 0) == 0) {
			options.seed = static_cast<unsigned>(std::stoul(require_arg_value(argc, argv, i, "--seed")));
		} else if(arg == "--time-limit" || arg.rfind("--time-limit=", 0) == 0) {
			options.time_limit_seconds = std::stod(require_arg_value(argc, argv, i, "--time-limit"));
			if(options.time_limit_seconds < 0) throw std::invalid_argument("--time-limit must be non-negative");
		} else if(arg == "--threads" || arg.rfind("--threads=", 0) == 0) {
			options.threads = std::max(1, std::stoi(require_arg_value(argc, argv, i, "--threads")));
		} else if(arg == "--help" || arg == "-h") {
			std::cerr << "Usage: solver [--seed UINT] [--time-limit SECONDS] [--threads N] < input.txt > solution.txt\n";
			std::exit(0);
		} else {
			throw std::invalid_argument("Unknown argument: " + arg);
		}
	}
	return options;
}

inline bool time_limit_reached(
	const std::chrono::steady_clock::time_point &start,
	double time_limit_seconds
) {
	if(time_limit_seconds == std::numeric_limits<double>::infinity()) return false;
	return std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count()
		>= time_limit_seconds;
}

inline int get_skill_id(
	std::unordered_map<std::string, int> &s2i,
	std::vector<std::vector<int>> &skill,
	std::vector<std::vector<int>> *s2c,
	const std::string &name
) {
	auto it = s2i.find(name);
	if(it != s2i.end()) return it->second;
	int id = static_cast<int>(s2i.size());
	s2i[name] = id;
	for(auto &levels : skill) levels.push_back(0);
	if(s2c) s2c->push_back(std::vector<int>());
	return id;
}

#endif
