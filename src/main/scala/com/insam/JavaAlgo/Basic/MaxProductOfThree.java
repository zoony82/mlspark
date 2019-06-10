package com.insam.JavaAlgo.Basic;

import java.util.Arrays;

public class MaxProductOfThree {
	public int solution(int[] arr) {
		Arrays.sort(arr);
		System.out.println(Arrays.toString(arr));
		int result;
		
		int arrSize= arr.length-1;
		// 모두 음수일때 : 제일큰거 세개 곱하면 됨
		if(arr[arrSize] < 0) {
			result = arr[arrSize] * arr[arrSize-1] * arr[arrSize-2];
		}
		
		// 첫번재가 양수이고, 세번째부터 모두 음수일때(두번째로 큰수 또는 세번째로 큰수가 음수일때) : 첫번째 양수와, 맨뒤 음수 두개 곱한거
		else if(arr[arrSize-1] < 0 || arr[arrSize-2] < 0) {
			result = arr[arrSize] * arr[0] * arr[1];
		}
		
		
		// 나머지, 큰수 3개가 모두 양수일때 :  큰수3개의 곱 or, 가장큰수*가장작은수*그다음작은수 중에 큰거
		else {
			int v1 = arr[arrSize] * arr[arrSize-1] * arr[arrSize-2];
			int v2 = arr[arrSize] * arr[0] * arr[1];
			
			if(v1 > v2) {
				result = v1;
			}
			else {
				result = v2;
			}
		}
				
		return result;
	}
}
