package com.insam.JavaAlgo;

import java.util.*;

public class postfixNotation{

	public static void main(String[] args) {
		// TODO Auto-generated method stub
		// 뒤에서 부터 사칙연산을 계산하기

		postfixNotation app = new postfixNotation();
		String input1 = "3 5 +";
		System.out.println(app.postCalc(input1));

		String input2 = "9 3 /";
		System.out.println(app.postCalc(input2));

		String input3 = "5 8 + 2 *";
		System.out.println(app.postCalc(input3));

		String input4 = "5 8 + 2 * 3 +";
		System.out.println(app.postCalc(input4));

		String input5 = "5 8 + 2 * 3 + 15 -";
		System.out.println(app.postCalc(input5));

	}

	int postCalc(String args) {
		String[] args_token = args.split("\\ ");
		Queue que = new LinkedList();

		int result = 0;

		// insert stack
		for( String str: args_token ){
			que.add(str);
		}

		// pop stack
		int preVal=0;
		int nextVal=0;
		while(!que.isEmpty())  {

			String val = (String)que.poll();

			if(val.equals("+")) {
				result = preVal + nextVal;
			} else if(val.equals("-")) {
				result = preVal - nextVal;
			} else if(val.equals("/")) {
				result = preVal / nextVal;
			} else if(val.equals("*")) {
				result = result * preVal;
			} else {
				preVal = Integer.parseInt(val);
				String val2 = (String)que.poll();
				// num? or calc?
				if(val2.equals("+")) {
					result = result + preVal;
				} else if(val2.equals("-")) {
					result = result - preVal;
				} else if(val2.equals("/")) {
					result = result / preVal;
				} else if(val2.equals("*")) {
					result = result * preVal;
				} else
					nextVal = Integer.parseInt(val2);
				}

			}

		return result;
	}

}
