//https://codeforces.com/contest/1848/problem/D
#include<bits/stdc++.h>

using i128 = __int128;
inline i128 max(i128 a, i128 b){return a > b ? a : b;}
inline i128 min(i128 a, i128 b){return a > b ? b : a;}

void println(i128 x){
	if(x == 0){
		std::cout << 0 << '\n';
		return ;
	}
    std::vector<int>v;
    while(x){
        v.push_back(x % 10);
        x /= 10;
    }
    for(int i = v.size() - 1; i >= 0; --i) std::cout << v[i];
    std::cout << '\n';
}

void solve(){
	int s, k;
    std::cin >> s >> k;
	
	if(s % 10 == 0) println((i128)s * k); 
	else if(s % 10 == 5) println(max((i128)s * k, (i128)(s + 5) * (k - 1)));
	else{
		i128 ans = 0;
		while(k > 0 and s % 10 != 2){
			ans = max(ans, (i128)k-- * s);
			s += s % 10;
		}
		auto fact = [&](i128 k, i128 a){
			i128 c[4];
			c[0] = ((5 * k - a) / 40.0 + 0.5);
			c[1] = ((5 * k - a - 7) / 40.0 + 0.5);
			c[2] = ((5 * k - a - 16) / 40.0 + 0.5);
			c[3] = ((5 * k - a - 29) / 40.0 + 0.5);
			for(int i = 0; i < 4; ++i){
				if(4 * c[i] + i > k) c[i] = (k - i) / 4 * 4 + i;
				else if(c[i] < 0) c[i] = min(k, i);
				else c[i] = c[i] * 4 + i;
			}
			c[0] = (k - c[0]) * (s + 5 * c[0]);
			c[1] = (k - c[1]) * (s + 5 * c[1] - 3);
			c[2] = (k - c[2]) * (s + 5 * c[2] - 4);
			c[3] = (k - c[3]) * (s + 5 * c[3] - 1);
		return max(max(c[0], c[1]), max(c[2], c[3]));
	};
		println(k == 0?ans:max(ans, fact(k, s)));
	}
}

signed main(){
    std::ios::sync_with_stdio(0);
    std::cin.tie(0);

	int T; 
    std::cin >> T;

    while(T--) solve();
	return 0;
}