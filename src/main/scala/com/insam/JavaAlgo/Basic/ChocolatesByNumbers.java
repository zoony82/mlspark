package com.insam.JavaAlgo.Basic;

public class ChocolatesByNumbers {
	public int solution(int N, int M) {
		int res = 0;
		res = maxDivide(N, M);
		if(res == 1) {
			return N;
		} else {
			return N/res;
		}
		
	}
	
	public int maxDivide(int a, int b) {
		if(a%b == 0) {
			return b;
		}
		else {
			return maxDivide(b, a%b);
		}
	}
}
