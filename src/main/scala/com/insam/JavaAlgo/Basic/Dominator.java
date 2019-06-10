package com.insam.JavaAlgo.Basic;

import java.util.HashMap;

public class Dominator {
	public int solution(int[] arr) {
		//int res=0;
		
		HashMap<Integer, Integer> hash = new HashMap<Integer, Integer>();
		for(int i=0 ; i<arr.length ; i++) {
			Integer temp = hash.get(arr[i]);
			if(temp == null) {
				temp = 0;
			}
			
			hash.put(arr[i], temp + 1);
			
			
			if(temp+1 > arr.length/2) {
				return i;
			}
		}
		
		
		
		return -1;
	}
}
