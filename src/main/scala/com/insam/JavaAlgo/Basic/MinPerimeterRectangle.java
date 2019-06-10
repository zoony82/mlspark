package com.insam.JavaAlgo.Basic;

import java.util.*;

public class MinPerimeterRectangle {
	public int solution(int A) {
	    
	    int minValue = Integer.MAX_VALUE;
	    
	    int sqrtNum = (int)(Math.sqrt(A));
	    
	    
	    for(int i=1 ; i<=sqrtNum ; i++) {
	    	//int iVal = 0;
	    	 
	    	if(A%i == 0) {
	    		int iVal = A/i;
	    		minValue = Math.min(minValue, (i + iVal)*2);
	    	}
	    	
	    }
	 
	    return minValue;
	}

}
