package com.insam.JavaAlgo.Basic;

import java.util.HashMap;
import java.util.Map;

public class OddOccurencesInArray {
	public int solution(int[] A) {
        // write your code in Java SE 8
     int rValue=0;
     
	 Map<Integer, Integer> tempMap = new HashMap<>();
     
	 // Array Insert
	 for(int i = 0; i < A.length ; i++) {
		 if(tempMap.get(A[i]) != null) {
			 tempMap.put(A[i], tempMap.get(A[i]) + 1);	 
		 } 
		 else {
			 tempMap.put(A[i], 1);
		 }
		 
		 
	 }
	 
	 //Odd Search
	 for(int key : tempMap.keySet()) {
		 if(tempMap.get(key)%2 == 1) {
			 rValue = key;
		 }
	 }
	 
	 
	 
	 return rValue;

	}
}
