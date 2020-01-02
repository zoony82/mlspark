package com.insam.JavaAlgo.codility;

import java.util.HashSet;
import java.util.Set;

/**
 * https://codility.com/programmers/lessons/4-counting_elements/frog_river_one/
 * @author abhishekkhare
 *
 */
public class FrogRiverOne {

	public static void main(String[] args) {
//		System.out.println(solution(5, new int[]{1,3,1,4,2,3,5,4}));
//		System.out.println(solution(10, new int[]{1,3,1,4,2,3,5,4}));
//		System.out.println(solution (4, new int[]{1,3,1,2,3,5,4}));
//		System.out.println(solution (1, new int[]{1,3,1,4,2,3,5,4}));
		//System.out.println(solution(4, new int[]{1,3,1,4,2,3,5,4,9,8,6,7,10,11}));
		//System.out.println(solution(10, new int[]{1,3,1,4,2,3,5,4,9,8,6,7,10,11}));
		System.out.println(solution(11, new int[]{1,3,1,4,2,3,5,4,9,8,6,7,10,11}));

	}

	public static int solution(int X, int[] A){
		Set<Integer> set = new HashSet<Integer>();
		for (int i = 0; i < A.length; i++) {
			System.out.println(A[i] +" --" + set);
			if(set.size()==X){
				return i-1;
			}else if(A[i]<=X){
				set.add(A[i]);
			}
		}
		if(set.size()==X){
			return A.length-1;
		}else{
			return -1;
		}
	}
}
