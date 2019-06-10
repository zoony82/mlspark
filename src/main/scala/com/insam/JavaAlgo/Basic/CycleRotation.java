package com.insam.JavaAlgo.Basic;

public class CycleRotation {
	public int[] solution(int[] A, int K) {
		//int[] ret=null;
		int[] ret;
		ret = new int[A.length];
		
		for(int i = 0; i < A.length ; i++) {
			//System.out.println(val);
			ret[(i+K) % A.length] = A[i];
		}
		
		return ret;
	}
}
