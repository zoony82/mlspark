package com.insam.JavaAlgo.codility;

import java.util.Arrays;
/**
 * https://codility.com/programmers/lessons/4-counting_elements/perm_check/
 * @author abhishekkhare
 *
 */
public class PermCheck {

	public static void main(String[] args) {
		System.out.println(solution(new int[]{1,2,3,5}));
		System.out.println(solution(new int[]{1,2,3,4,5}));
		System.out.println(solution(new int[]{2,3,4,5}));
		System.out.println(solution(new int[]{2,3,4,1,6}));

	}

	public static int solution(int[] A){
		Arrays.sort(A);
		for (int i = 0; i < A.length; i++) {
			if(A[i]!=i+1){
				return 0;
			}
		}
		return 1;
	}
}
