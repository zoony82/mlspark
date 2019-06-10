package com.insam.JavaAlgo.Basic;

public class CountDiv {
	public int solution(int A, int B, int K) {
		int result=0;
		/*
		for(int i = A; i<B; i++) {
			int temp = (i)%K;
			if(temp == 0) {
				result++;
			}
		}
		
		
		return result;
		*/
		//Math.ceil 올림 (6, 11, 2)
		System.out.println(A % K);
		System.out.println(K - (A % K));
		System.out.println((K - (A % K)) % K);
		System.out.println((K - (A % K)) % K);
		return  (int) Math.ceil((B - A + 1 - (K - (A % K)) % K) / (double) K);
	}
}
