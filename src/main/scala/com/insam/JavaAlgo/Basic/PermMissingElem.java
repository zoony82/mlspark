package com.insam.JavaAlgo.Basic;

public class PermMissingElem {
	public int solution(int[] A) {
	    boolean[] checked = new boolean[A.length + 2];
	 
	    for (int i = 0; i < A.length; i++) {
	        checked[A[i]] = true;
	    }
	 
	    for (int i = 1; i < checked.length; i++) {
	        if (!checked[i]) {
	            return i;
	        }
	    }
	 
	    return -1;
	}

}
