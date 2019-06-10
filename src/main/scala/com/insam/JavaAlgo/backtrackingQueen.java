package com.insam.JavaAlgo;

import java.util.Arrays;

// http://www.jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=1162&sca=3030
// 체스 판에서 퀸을 적당한데 위치시키기
public class backtrackingQueen {
    public static int N =4;
    public static int count = 0;
    public static int [] col = new int[N+1];

    public static void main(String args[]) {
        System.out.println("queen");


        setQueen(0, col, N+1);

        System.out.println(count);
    }

    public static void setQueen(int i, int[] c, int n) {
        if(check(i,c)) {
            if(i==n-1) {
                count++;
                System.out.println(Arrays.toString(c));
            }
            else {
                // 만약 같으면 복사 하나 뜨자.
                int[] col = Arrays.copyOf(c, n);
                for(int j=1; j<n;j++) {
                    col[i+1] = j;
                    setQueen(i+1, col, n);
                }
            }

        }
    }

    public static boolean check(int i, int[] c) {
        for(int k=1; k<i; k++) {
            if(c[i] == c[k] || Math.abs(c[i] - c[k]) == i-k) {
                return false;
            }
        }
        return false;
    }
}
