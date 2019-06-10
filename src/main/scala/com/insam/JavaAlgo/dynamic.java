package com.insam.JavaAlgo;

import java.util.Arrays;

public class dynamic {
    // 1인당 17kg만 담을수 있는 가방에 최대한 가치가 높게 가져가기
    public static void main(String args[]){
        int N = 5; //짐 개수
        int M = 17; // 가용 무게
        int[] k = new int[N+1]; // 무게
        int[] v = new int[N+1]; // 가치
        k[1] = 3;
        k[2] = 4;
        k[3] = 7;
        k[4] = 8;
        k[5] = 9;

        v[1] = 4;
        v[2] = 5;
        v[3] = 10;
        v[4] = 11;
        v[5] = 13;

        // 메모리 만들기
        int[][] dp = new int[N+1][M+1];

        for(int i=1; i<=N ; i++){ //짐 개수 5
            for(int w=1; w<=M ; w++){ // 키로수 17
                if(k[i] <= w){

                    // dp[i-1][w] : 이전 짐의, 해당 키로 제한에 대한 가치
                    // v[i] : 해당 짐의 가치
                    // dp[i-1][w-k[i]] : 이전 짐의, 제한 무게-해당 키로수에 대한 합
                                    //ex)  9kg 넣을때 12kg으로 제한까지 왔다면 12-9 이므로 3, 이전 8kg에서 3에 대한 가치는 4이다.
                                    // 이때 이전 8kg의 16(dp[i-1][w])과 13+4=17 중 큰 값을 넣는다.
                    // dp[i-1][w-k[i]] : dp[이전 가방의 최대가치][현재 무게 - 해당 가방 무게 = 추가로 담을 수 있는 무게]
                    dp[i][w] = Math.max(dp[i-1][w], v[i]+dp[i-1][w-k[i]]);

                }else {
                    dp[i][w] = dp[i-1][w]; //위에것 그대로 가져오기
                }
            }
            System.out.println(Arrays.toString(dp[i]));
        }
        System.out.println(dp[N][M]);
    }
}
