// Copyright (C) 2026, Coudert--Osmont Yoann
// SPDX-License-Identifier: MIT
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <unordered_set>
#include <vector>

using namespace std;

int main() {
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int C, P;
	cin >> C >> P;
	vector<string> Cnames(C), Pnames(P);
	unordered_map<string, int> s2i;
	vector<array<int, 800>> skill(C);
	vector<int> ts(C, 0);
	vector<vector<pair<int, int>>> req(P);
	vector<int> D(P), S(P), B(P);

	for(int c = 0; c < C; ++c) {
		int n;
		cin >> Cnames[c] >> n;
		skill[c].fill(0);
		while(n--) {
			string s; int l;
			cin >> s >> l;
			if(!s2i.count(s)) s2i[s] = s2i.size();
			skill[c][s2i[s]] = l;
			ts[c] += l;
		}
	}
	for(int p = 0; p < P; ++p) {
		int r;
		cin >> Pnames[p] >> D[p] >> S[p] >> B[p] >> r;
		while(r--) {
			string s; int l;
			cin >> s >> l;
			if(!s2i.count(s)) s2i[s] = s2i.size();
			req[p].emplace_back(s2i[s], l);
		}
	}

	int score = 0;
	vector<pair<int, vector<int>>> sol;

	vector<int> av(C, 0);
	vector<int> order(P);
	vector<bool> chosen(C, false);
	iota(order.begin(), order.end(), 0);
	mt19937 mt;
	bool need_lvl = false;
	const auto ckey = [&](int c, int m, int s, int l) { return make_tuple(max(av[c], m), skill[c][s] <= l ? 0 : 1, skill[c][s] <= l ? -skill[c][s] : skill[c][s]-l, ts[c], av[c], c); };
	mt.seed(2654435761u);
	const auto skill0 = skill;
	const auto ts0 = ts;
	int best_score = -1;
	vector<pair<int, vector<int>>> best_sol;
	mt.seed(12345);
	vector<int> perturb(P, 0), best_perturb(P, 0);
	const int NITER = 2200;
	for(int iter = 0; iter < NITER; ++iter) {
	if(iter > 0) {
		perturb = best_perturb;
		for(int k = 0; k < P/10; ++k) { int p = mt() % P; perturb[p] = (int)(mt() % 13) - 6; }
	}
	skill = skill0; ts = ts0;
	fill(av.begin(), av.end(), 0);
	fill(chosen.begin(), chosen.end(), false);
	score = 0; sol.clear();
	need_lvl = false;
	order.resize(P); iota(order.begin(), order.end(), 0);
	while(true) {
		bool bad = true;
		vector<int> fail;
		if(need_lvl) sort(order.begin(), order.end(), [&](int a, int b) { int ka=S[a]+5*perturb[a], kb=S[b]+5*perturb[b]; long long ra=(long long)ka*D[b], rb=(long long)kb*D[a]; if(ra != rb) return ra > rb; if(S[a] != S[b]) return S[a] > S[b]; if(B[a]-D[a] != B[b]-D[b]) return B[a]-D[a] > B[b]-D[b]; if(D[a] != D[b]) return D[a] < D[b]; return a < b; });
		else sort(order.begin(), order.end(), [&](int a, int b) { int ka=B[a]-D[a]+6*perturb[a], kb=B[b]-D[b]+6*perturb[b]; if(ka != kb) return ka < kb; long long ra=(long long)(S[a]+perturb[a])*D[b], rb=(long long)(S[b]+perturb[b])*D[a]; if(ra != rb) return ra > rb; if(S[a] != S[b]) return S[a] > S[b]; if(D[a] != D[b]) return D[a] < D[b]; if(B[a] != B[b]) return B[a] < B[b]; return a < b; });
		for(int p : order) {
			int mav = 0;
			vector<int> cs(req[p].size(), -1);
			static array<int, 800> ms; for(auto [s,l] : req[p]) ms[s] = 0;
			vector<int> ridx(req[p].size());
			iota(ridx.begin(), ridx.end(), 0);
			sort(ridx.begin(), ridx.end(), [&](int a, int b){ int la=req[p][a].second, lb=req[p][b].second; if(la != lb) return la > lb; return a < b; });
			int cnt = 0;
			for(int idx : ridx) {
				auto [s, l] = req[p][idx];
				if(ms[s] >= l) --l;
				int best = -1;
				for(int c = 0; c < C; ++c) if(!chosen[c] && skill[c][s] >= l)
					if(best == -1 || ckey(c, mav, s, l) < ckey(best, mav, s, l))
						best = c;
				if(best == -1) break;
				cs[idx] = best;
				++cnt;
				for(auto [s2,l2] : req[p]) ms[s2] = max(ms[s2], skill[best][s2]);
				chosen[best] = true;
				mav = max(mav, av[best]);
			}
			for(int c : cs) if(c != -1) chosen[c] = false;
			int end = mav+D[p];
			if(cnt < (int)req[p].size()) {
				fail.push_back(p);
				continue;
			}
			for(int i = 1; i < cs.size(); ++i) if(skill[cs[i]][req[p][i].first] > req[p][i].second)
				for(int j = 0; j < i; ++j) if(skill[cs[j]][req[p][j].first] > req[p][j].second)
					if(skill[cs[i]][req[p][j].first] >= req[p][j].second-1 && skill[cs[j]][req[p][i].first] >= req[p][i].second-1)
						swap(cs[i], cs[j]);
			bool impLvl = false;
			for(int i = 0; i < cs.size(); ++i) {
				av[cs[i]] = end;
				auto [s, l] = req[p][i];
				if(skill[cs[i]][s] <= l) {
					++ skill[cs[i]][s];
					++ ts[cs[i]];
					impLvl = true;
				}
			}
			if(!impLvl && end >= B[p]+S[p]) {
				for(int i = 0; i < (int)cs.size(); ++i) av[cs[i]] -= 0;
				fail.push_back(p);
				continue;
			}
			score += max(0, S[p] - max(0, end-B[p]));
			sol.emplace_back(p, cs);
			bad = false;
		}
		if(bad) {
			if(need_lvl) break;
			need_lvl = true;
		}
		order = fail;
	}
	if(score > best_score) { best_score = score; best_sol = sol; best_perturb = perturb; }
	}
	score = best_score; sol = best_sol;

	cerr << score << endl;
	cout << sol.size() << endl;
	for(const auto &[p, cs] : sol) {
		cout << Pnames[p] << '\n';
		for(int c : cs) cout << Cnames[c] << ' ';
		cout << '\n';
	}

	return 0;
}