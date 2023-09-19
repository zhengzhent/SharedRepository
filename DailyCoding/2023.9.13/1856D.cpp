//https://codeforces.com/contest/1856/problem/D

#include<bits/stdc++.h>

int ask(int a, int b){
    std::cout << '?' << ' '<< a << ' ' << b << std::endl;
    int mid_;
    std::cin >> mid_;
    return mid_;
}

int dfs(int l, int r){
    if(l == r) return l;
    if(l + 1 == r){
        if(ask(l, r)) return l;
        else return r;
    }
    int mid = l + r >> 1;
    int ll = dfs(l, mid), rr = dfs(mid + 1, r);
    int x = ask(l, rr - 1), y = ask(l, rr);
    if(x == y) return rr;
    else return ll;
};

int main() {
    int T;
    std::cin >> T;
    while(T--) {
        int n;
        std::cin >> n;
        int ans = dfs(1, n);
        std::cout << '!' << ' ' << ans << std::endl;
    }
    return 0;
}