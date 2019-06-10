package com.insam.JavaAlgo.Basic;

public class BinaryGap {
	public int solution(int N) {
		
		int maxGap = 0;
		
		String biStr = Integer.toBinaryString(N);
		
		System.out.println("binary " + biStr);
		
		int firstOneIndex = getNextOneIndex(biStr, 0);
		if(firstOneIndex < 0) {
			return 0;
		}
		
		
		
		while(true) {
			int NextOneIndex = getNextOneIndex(biStr, firstOneIndex + 1);
			
			if(NextOneIndex < 0) {
				break;
			}
			else if(maxGap < (NextOneIndex - firstOneIndex-1)) {
				maxGap = (NextOneIndex - firstOneIndex-1);
			}
			firstOneIndex = NextOneIndex;
		}
		
		return maxGap;
	}
	
	private int getNextOneIndex(String bitStr, int startIndex) {
		int nextOneIndex = -1;
		
		for(int i=startIndex ; i < bitStr.length() ; i++) {
			char c = bitStr.charAt(i);
			if(c == '1') {
				nextOneIndex = i;
				break;
			}
		}
		
		return nextOneIndex;
	}
}
