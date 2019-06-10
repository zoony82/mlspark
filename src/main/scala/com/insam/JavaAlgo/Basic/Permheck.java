package com.insam.JavaAlgo.Basic;

import java.util.*;

public class Permheck {
	public int solution(int[] A) {
	    Set<Integer> marks = new HashSet<>();
	 
	    for (int i = 0; i < A.length; i++) {
	        if (A[i] > A.length) {
	            return 0;
	        }
	 
	        if (marks.contains(A[i])) {
	            return 0;
	        }
	 
	        marks.add(A[i]);
	    }
	 
	    return 1;
	}

}
