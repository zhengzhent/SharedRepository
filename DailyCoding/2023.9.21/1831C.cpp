//https://codeforces.com/contest/1831/problem/C
#include<bits/stdc++.h>

class node{
public:
    int u, i, k;
};

void solve() {
    int n;
    std::cin >> n;
    std::vector<std::vector<std::pair<int, int>>> a(n + 1);
    for(int i = 1; i < n; ++i) {
        int u, v;
        std::cin >> u >> v;
        a[u].push_back({v, i});
        a[v].push_back({u, i});
    }
    std::queue<node> q;
    std::vector<bool> vis(n + 1);
    q.push({1, 0, 1});
    int ans = 1;
    while(!q.empty()) {
        auto [u, i, k] = q.front();
        q.pop();
        if(vis[u]) continue;
        ans = std::max(ans, k);
        vis[u] = true;
        for(auto [it1, it2]:a[u]) {
            if(it2 < i) q.push({it1, it2, k + 1});
            else q.push({it1, it2, k});
        }
    }
    std::cout << ans << '\n';
}

int main() {
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    int T;
    std::cin >> T;
    while(T--) solve();
    return 0;
}