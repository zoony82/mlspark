package com.insam.JavaAlgo.Basic;

import java.util.*;

public class MaxProfit {
	public int solution(int[] A) {
	    int maxProfit = 0;
	    int minVal = 9000000;
	    
	    minVal = A[0];
	    for(int i=1 ; i<A.length ; i++) {
	    	minVal = Math.min(minVal, A[i]);
	    	maxProfit = Math.max(maxProfit,A[i] - minVal);
	    }
	 
	    return maxProfit;
	}

}
