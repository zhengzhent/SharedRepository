//https://codeforces.com/contest/1791/problem/F
#include<bits/stdc++.h>

const int N = 2e5 + 5;

class DSU {
private:
    int f[N];
public:
    void init(int n) {for(int i=1;i<=n;i++)f[i]=i;}
    int find(int x){return x==f[x]?x:f[x]=find(f[x]);}
    bool same(int x,int y){return find(x)==find(y);}
    void merge(int x,int y) {if(!same(x,y))f[find(x)]=find(y);}
}T;

int fact(int x){
    int sum = 0;
    while(x){
        sum += x % 10;
        x /= 10;
    }
    return sum;
}

void solve() {
    int n, q;
    std::cin >> n >> q;
    T.init(n + 1);
    std::vector<int>a(n + 1);
    for(int i = 1; i <= n; ++i) std::cin >> a[i];
    while(q--){
        int op;
        std::cin >> op;
        if(op == 1){
            int l, r;
            std::cin >> l >> r;
            for(int i = l; i <= r; i = T.find(i + 1)){
                a[i] = fact(a[i]);
                if(a[i] <= 9) T.merge(i, i + 1);
            }
        }else{
            int x;
            std::cin >> x;
            std::cout << a[x] << '\n';
        }
    }
    
}

int main(){
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);
    int T;
    std::cin >> T;
    while(T--) solve();
    return 0;
}