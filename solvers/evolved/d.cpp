// Copyright (C) 2026, Coudert--Osmont Yoann
// SPDX-License-Identifier: MIT
// Licensed under the MIT License. See LICENSE file in the project root for full license information.

#include <algorithm>
#include <array>
#include <iostream>
#include <numeric>
#include <random>
#include <unordered_map>
#include <vector>

#include "../mtp_runtime.hpp"

using namespace std;
constexpr int NS = mtp::NS;  // see ../mtp_runtime.hpp

int main(int argc, char **argv) {
	mtp::start_timer();
	const unsigned seed = mtp::parse_seed(argc, argv, true);
	ios::sync_with_stdio(false);
	cin.tie(nullptr);

	int C, P;
	cin >> C >> P;
	vector<string> Cnames(C), Pnames(P);
	unordered_map<string, int> s2i;
	vector<array<int, NS>> skill(C);
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
			mtp::skill_id(s2i, s);  // exits if > NS distinct skills
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
			mtp::skill_id(s2i, s);  // exits if > NS distinct skills
			req[p].emplace_back(s2i[s], l);
		}
	}
	cerr << C << ' ' << P << ' ' << s2i.size() << endl;

	long long score = 0;  // total score can exceed INT_MAX on some instances
	vector<pair<int, vector<int>>> sol;

	vector<int> av(C, 0);
	vector<int> order(P);
	vector<bool> chosen(C, false);
	iota(order.begin(), order.end(), 0);
	mt19937 mt(seed);
	int CKEYMODE = 0;
	const auto ckey = [&](int c, int m, int s, int l) { return CKEYMODE==1 ? make_tuple(max(av[c], m), ts[c], skill[c][s]-l) : CKEYMODE==2 ? make_tuple(max(av[c], m), skill[c][s]-l, -ts[c]) : CKEYMODE==3 ? make_tuple(max(av[c], m), av[c], skill[c][s]-l) : CKEYMODE==4 ? make_tuple(max(av[c], m), -(skill[c][s]-l), ts[c]) : CKEYMODE==5 ? make_tuple(max(av[c], m), skill[c][s]-l, av[c]) : make_tuple(max(av[c], m), skill[c][s]-l, ts[c]); };
	long long bestscore = -1;
	vector<pair<int, vector<int>>> bestsol;
	auto skill0 = skill;
	auto ts0 = ts;
	constexpr int NRESTARTS = 76;  // restart count unchanged; restart r uses seed_seq{seed, r}
	const int latew[] = {4, 1, 8, 16, 2, 0, 3, 6, 12, 24, 1, 5};
	const int mavw[] = {64, 16, 256, 8, 128, 32, 96, 48, 192, 24, 384, 512, 64, 1024, 4, 768, 12, 160, 2048, 6, 320, 80, 640, 20};
	int sidx = 0;
	bool stop = false;
	for(int restart = 0; restart < NRESTARTS && !stop; ++restart) {
	int LATEW = latew[sidx % 12];
	int MAVW = mavw[sidx % 24];
	int LVLW = (sidx % 3 == 0) ? 0 : (sidx % 3 == 1) ? 1 : 4;
	CKEYMODE = (sidx >> 3) % 6;
	int NTEST = (sidx >= 60) ? 20 + (sidx % 25) : 30 + (sidx % 16);
	sidx++;
	{ seed_seq restart_seq{seed, static_cast<unsigned>(restart)}; mt.seed(restart_seq); }
	skill = skill0; ts = ts0;
	score = 0;
	sol.clear();
	fill(av.begin(), av.end(), 0);
	fill(chosen.begin(), chosen.end(), false);
	order.resize(P);
	iota(order.begin(), order.end(), 0);
	while(true) {
		bool bad = true;
		vector<int> fail;
		sort(order.begin(), order.end(), [&](int a, int b) {
			if(sidx&32) {
				long long ka=(long long)S[a]*D[b]*req[b].size(), kb=(long long)S[b]*D[a]*req[a].size();
				if(ka!=kb) return kb<ka;
			}
			if(sidx&8) { if(B[a] != B[b]) return B[a] < B[b]; }
			if(sidx&4) { if(S[a] != S[b]) return S[a] > S[b]; }
			if(sidx%6==5) { if(B[a]+S[a] != B[b]+S[b]) return B[a]+S[a] < B[b]+S[b]; }
			if((long long)S[a]*D[b] != (long long)S[b]*D[a]) return (long long)S[b]*D[a] < (long long)S[a]*D[b];
			if(B[a] != B[b]) return (sidx&1) ? B[a] > B[b] : B[a] < B[b];
			return (sidx&2) ? req[a].size() < req[b].size() : req[a].size() > req[b].size();
		});
		for(int p : order) {
			if(mtp::time_up()) { stop = true; break; }  // 590 s: stop, keep committed projects
			int beste = -1; long long besto = -1;  // oo/besto widened: MAVW*(1000000-mav) can overflow int
			vector<int> cs(req[p].size(), 0), cs2, ro=cs;
			iota(ro.begin(), ro.end(), 0);
			array<int, NS> ms;
			vector<int> usk;
			for(auto &pr : req[p]) usk.push_back(pr.first);
			sort(usk.begin(), usk.end());
			usk.erase(unique(usk.begin(), usk.end()), usk.end());
			for(int test = 0; test < NTEST; ++test) {
				if(mtp::time_up()) break;
				shuffle(ro.begin(), ro.end(), mt);
				for(int si : usk) ms[si] = 0;
				int mav = 0, nr = req[p].size(); long long oo = 0;
				for(int o : ro) {
					auto [s, l] = req[p][o];
					const int l0 = l;
					if(ms[s] >= l && (sidx%4 != 3)) --l;
					int best = -1, best2 = -1;
					for(int c = 0; c < C; ++c) if(!chosen[c] && skill[c][s] >= l)
						if(best == -1 || ckey(c, mav, s, l0) < ckey(best, mav, s, l0)) {
							best2 = best; best = c;
						} else if(best2 == -1 || ckey(c, mav, s, l0) < ckey(best2, mav, s, l0))
							best2 = c;
					if(best == -1) break;
					(void)best2;
					-- nr;
					cs[o] = best;
					if(skill[best][s] <= l0) oo += 1 + LVLW;
					for(int si : usk) ms[si] = max(ms[si], skill[best][si]);
					chosen[best] = true;
					mav = max(mav, av[best]);
				}
				for(int c : cs) chosen[c] = false;
				int end = mav+D[p];
				if(nr) continue;
				if(end >= B[p]+S[p] && sidx < 60) continue;
				oo -= LATEW*max(0, end-B[p]);
				if(sidx&64) oo += S[p]*max(0,(int)req[p].size()-nr);
				if(sidx%5==4) oo += MAVW*min(50, max(0, B[p]-end));
				if(!(sidx&256))
				for(int i = 1; i < cs.size(); ++i) if(skill[cs[i]][req[p][i].first] > req[p][i].second)
					for(int j = 0; j < i; ++j) if(skill[cs[j]][req[p][j].first] > req[p][j].second)
						if(skill[cs[i]][req[p][j].first] >= req[p][j].second-1 && skill[cs[j]][req[p][i].first] >= req[p][i].second-1) {
							swap(cs[i], cs[j]);
							oo += 2;
						}
				oo += (long long)MAVW*(1000000-mav);
				if(oo > besto) {
					besto = oo;
					beste = end;
					cs2 = cs;
				}
			}
			int end = beste;
			int gain = S[p] - max(0, end-B[p]);
			if(cs2.empty() || (gain <= 0 && !(sidx&1))) {
				fail.push_back(p);
				continue;
			}
			cs = cs2;
			score += S[p] - max(0, end-B[p]);
			for(int i = 0; i < cs.size(); ++i) {
				av[cs[i]] = end;
				auto [s, l] = req[p][i];
				if(skill[cs[i]][s] <= l) {
					++ skill[cs[i]][s];
					++ ts[cs[i]];
				}
			}
			sol.emplace_back(p, cs);
			bad = false;
		}
		if(stop || bad) break;
		order = fail;
	}
	if(score > bestscore) { bestscore = score; bestsol = sol; }
	}
	score = bestscore;
	sol = bestsol;

	cerr << score << endl;
	cout << sol.size() << endl;
	for(const auto &[p, cs] : sol) {
		cout << Pnames[p] << '\n';
		for(int c : cs) cout << Cnames[c] << ' ';
		cout << '\n';
	}

	return 0;
}