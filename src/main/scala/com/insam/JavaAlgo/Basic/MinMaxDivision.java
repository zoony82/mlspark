package com.insam.JavaAlgo.Basic;

public class MinMaxDivision {
	public int solution(int K, int M, int[] A) {
		// 최대수, 최소수를 구하자.
		// 그런다음 중간값  정해서 +1, -1 하면서  그  수를 넘게되는 배열이 K 보다 작은 최소수를 찾자
		int min=0;
		int max=0;
		int mid=0;
		for(int i = 0; i<A.length; i++) {
			min = Math.max(min, A[i]);
			max = max + A[i];
		}
		
		
		System.out.println("min , max , mid : " + min + "," + max+ "," + mid);
		int result=0;
		while(min<=max) {
			mid = (max+min)/2;
			
			// k 개만큼 나뉘어지는지 확인
			if(solidCheck(mid, K, A)) {
				max = mid-1;
				result = mid;
			} else {
				min = mid +1;
			}
		}
		return result;
	}
	
	public boolean solidCheck(int mid, int k, int[] A) {
		int sum=0;
		int k_cnt = 0;
		for(int i=0; i<A.length; i++) {
			sum = sum + A[i];
			if(sum>mid) {
				k_cnt++;
				sum = A[i]; // 다음 K 부터 계산 해야 하니 넘겨준다.
			}
			if(k_cnt>k-1) {
				return false;
			}
		}
		return true;
	}
	
}
