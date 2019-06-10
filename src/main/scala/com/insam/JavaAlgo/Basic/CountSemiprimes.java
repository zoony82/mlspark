package com.insam.JavaAlgo.Basic;

import java.util.Arrays;
import java.util.HashMap;

public class CountSemiprimes {
	public int[] solution(int N, int[] P, int[] Q) {
		//int res=0;
		
		int[] result = new int[P.length]; // default 0
		int[] flags = new int[N+1];  // default 0
		int[] presum = new int[N+1];  // default 0
		
		// 0 : prime -> 2,3,5,7,9,,,
		// 1 : No prime -> 0, 1,,,
		// 2 : semi prime -> 4,5,8,10,,		
		flags[0] = 1;
		flags[1] = 1;
		
		// 해당 수의 제곱부터 시작해서 배수를 더하며 검색해서, no prime 처리
		// prime : 2,3,5,7,11,,
		// no prime : 1,4,6,8,9,,
		for(int i=2; i*i <= N ; i++) {
			if(flags[i] == 1) {
				continue;
			}
			
			int k = i*i;
			while(k<=N) {
				flags[k] = 1;
				k = k+i;
			}
		}
		//System.out.println(Arrays.toString(flags));
		
		// 당 수의 제곱부터 시작해서 배수를 더하며 검색해서, semi prime 처리
		// prime 중에서, 나누어지는 수도 prime인 경우
		for(int i=2; i*i <=N ; i++) {
			if(flags[i] == 1) {
				continue;
			}
			
			int k = i*i;
			while(k<=N) {
				if(flags[i] == 0 && flags[k/i]==0) {
					flags[k] = 2; // semi prime	
				}
				k = k+i;
			}
		}
		
		//System.out.println(Arrays.toString(flags));
		
		// presum 누적치 미리 계싼
		int sum = 0;
		for(int i=0 ; i<=N ; i++) {
			if(flags[i]==2){
				sum++;
			}
			presum[i] = sum;
		}
		
		//System.out.println(Arrays.toString(presum));
		
		for(int i=0 ; i<P.length ; i++) {
			result[i] = presum[Q[i]] - presum[P[i]-1];
		}
		
		
		
		
		return result;
	}
}
