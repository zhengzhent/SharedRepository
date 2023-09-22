//https://codeforces.com/problemset/problem/1841/D
#include<bits/stdc++.h>

struct node {
    int l, r;
    friend bool operator< (node a, node b) {
        return a.r < b.r;
    }
};

void solve() {
    int n;
    std::cin >> n;
    std::vector<node> v(n), t;
    for(int i = 0; i < n; ++i) std::cin >> v[i].l >> v[i].r;
    std::sort(v.begin(), v.end());

    for(int i = 0; i < n; ++i) {
        for(int j = i + 1; j < n; ++j) {
            if(v[i].r >= v[j].l) t.push_back({std::min(v[i].l, v[j].l), v[j].r});
            else break;
        }
    }
    std::sort(t.begin(), t.end());
    int ans = 0, r = -1;
    for(auto [ll, rr]:t) if(ll > r) r = rr, ++ans;
    std::cout << n - 2 * ans << '\n';
}

int main() {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    int T;
    std::cin >> T;
    while(T--) solve();
    return 0;
}