package com.insam.JavaAlgo.Basic;

import java.util.Arrays;

public class AlgoRun {
	
	
	public static void main(String[] args) {
		
		
		int intResult;
//
//
//		// 1. Iterations
//		//양의정수를 받아서 바이너리열로 변환 했을 때, 가장 긴 1 사이의 갭을 구해라.
//		System.out.println("BinaryGap");
//		BinaryGap binary = new BinaryGap();
//		intResult = binary.solution(3015);
//		System.out.println(intResult);

//		// 2. Arrays
//		// 정수의 배열에서 홀수로 존재하는 정수를 찾아내라
//		System.out.println("OddOccurencesInArray");
//		OddOccurencesInArray odd = new OddOccurencesInArray();
//		int [] array1 = {9, 3, 9, 3, 9, 7, 9, 8};
//		intResult = odd.solution(array1);
//		System.out.println(intResult);
//
//		// 2. Arrays
//		// 정수의 배열을 K만큰 오른쪽으로 돌려라
//		CycleRotation cycleRotate = new CycleRotation();
//		int[] temp = {3,4,5,6,7,8,9,10,11,12};
//		int[] retArray = cycleRotate.solution(temp,3);
//		System.out.print(Arrays.toString(retArray));
//

//
//		// 3. Time Complexity
		// 1부터 시작되는 정수들의 배열에서, 딱 1개 숫자가 등장하지 않음, 그걸 찾아라
//		PermMissingElem missing = new PermMissingElem();
//		int[] temp = {2,3,5,6,1};
//		int ret = missing.solution(temp);
//		System.out.print(ret);
//
		// 3. Time Complexity
		// 정수 배열을 두 구간으로 쪼갯을 때, 구간별 합의 차가 가장 적은 곳 찾기
//		TapeEquilbrium tape = new TapeEquilbrium();
//		int[] temp = {2,3,5,6,1};
//		int ret = tape.solution(temp);
//		System.out.print(String.valueOf(ret));

//		// 3. Time Complexity
//
//		// 4. Counting Elements
		// 정수 배열이 1부터 존재할때, 순열인지 확인하기
//		Permheck perm = new Permheck();
//		int[] temp = {2,3,5,6,1,4};
//		int ret = perm.solution(temp);
//		System.out.print(String.valueOf(ret));
//
//		// 5. Prefix Sums
//		// A와 B사이에서 K로 나누어 떨어지는 수가 몇개인지 찾아라
//		CountDiv cd = new CountDiv();
//		int res = cd.solution(6, 11, 2);
//		System.out.println(res);

//		// 6. Sorting
//		// 정수의 배열에서 3개를 골라서 곱했을때 가장 큰 수를 구해라
//		MaxProductOfThree mpo = new MaxProductOfThree();
//		int[] arr = {3,7,9,-10,4,5};
//		int[] arr2 = {-3,-7,-9,-10,-4,-5};
//		int[] arr3 = {-3,-7,-9,-10,4,-5};
//		int result = mpo.solution(arr);
//		int result2 = mpo.solution(arr2);
//		int result3 = mpo.solution(arr3);
//		System.out.println(result);
//		System.out.println(result2);
//		System.out.println(result3);
//
//		// 7. Stacks and Queues
//		// 이전 블록이 없거나, 블록의 높이가 다른 경우마다 새로운 블록이 필요해진다
//		// 이전 블록이 이후에 나타나는 블록의 높이보다 높은 경우, 그 블록은 별개의 블록으로 처리하여야 하므로 stack에서 제거해준다.
//
//		StoneWall stw = new StoneWall();
//		int[] arr = {8, 8, 5, 7, 9, 8, 7, 4, 8};
//		int res = stw.solution(arr);
//		System.out.println(res);
//
//
//		// 8. Leader
//		// Dominator
//		// Find an index of an array such that its value occurs at more than half of indices in the array.
//		// 그 값이 배열의 인덱스의 절반 이상에서 발생하도록 배열의 인덱스를 찾습니다.
//		Dominator dom = new Dominator();
//		int[] arr = {3, 4, 3, 2, 3, -1, 3, 3};
//		int res = dom.solution(arr);
//		System.out.println(res);
//
//
//		// 9. Maximum slice problem
		// MaxProfit
		// Given a log of stock prices compute the maximum possible earning.
		// 주식을 언제 사서 언제 팔게 되는데 이때 낼 수 있는 최대 이익은 얼마인가

//		MaxProfit mpf = new MaxProfit();
//		int[] arr = {23171, 21011, 21123, 21366, 21013, 21367};
//		int temp = mpf.solution(arr);
//		System.out.println(temp);

//
//
//
//		// 10. Prime and composite numbers
//		// Min Perimeter(둘레) Rectangle
//		// Find the minimal perimeter of any rectangle whose area equals N.
//		//사각형의 넓이 N 이 주어지고 사각형의 둘레의 최소 길이를 구하는 문제이다.
//		// 즉 A * B = N 일때 2 * (A + B) 가 최소가 되는 값을 구하면 된다.
//		MinPerimeterRectangle mpr = new MinPerimeterRectangle();
//		int temp = mpr.solution(30);
//		System.out.println(temp);
//
//
//		// 11. Sieve of Eratosthenes(체)
//		// CountSemiprimes
//		// count the semiprime numbers in the given range [a..b]
//		//(에라토스테네스의) 체처럼 숫자들을 모두 체 위에 놓고 소수가 아닌 수들을 걸러내면 소수만 남는다.
//		CountSemiprimes csp = new CountSemiprimes();
//		int N = 26;
//		int[] P = {1,4,6};
//		int[] Q = {26,10,20};
//		int[] temp = csp.solution(N, P, Q);
//
//
//
//		// 12. Euclidean algorithm
//		// ChocolatesByNumbers
//		// There are N chocolates in a circle. Count the number of chocolates you will eat.
//		// 최대 공약수를 이용하는 문제로, 최대공약수가 1이 나오면, 초코렛을 다 먹는거고, 아닐경우, 초기 N값을 최대공약수로 나눈 값이 먹는 초코렛 수 이다.
//		ChocolatesByNumbers cbn = new ChocolatesByNumbers();
//		int res = cbn.solution(10, 4);
//		System.out.println(res);
//
//
//
//		// 14. Binary Search Algorithm
//		// MinMaxDivision
//		// Divide array A into K blocks and minimize the largest sum of any block.
//		MinMaxDivision mmd = new MinMaxDivision();
//		int[] arr = {2, 1, 5, 1, 2, 2, 2};
//
//		int res = mmd.solution(3, 5, arr);
//		System.out.println(res);
//
//
//		// 15. Dynamic Programming
//		// NumberSolitaire
//		// In a given array, find the subset of maximal sum in which the distance between consecutive elements is at most 6.
//
//		int[] arr = {1, -2, 0, 9, -1, -2};
//		NumberSolitaire ns = new NumberSolitaire();
//		int a = ns.solution(arr);
//		System.out.println(a);

		// kakao 추석 트래픽


				
	}
	 
	
	
}
