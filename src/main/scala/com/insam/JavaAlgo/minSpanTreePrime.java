package com.insam.JavaAlgo;

import java.util.ArrayList;
import java.util.List;

public class minSpanTreePrime{

    public static void main(String[] args) {
        // TODO Auto-generated method stub
        //http://www.jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=340&sca=3020

        int N=5;
        int[][] ia= {{0,  0, 0,  0, 0, 0},
                {0,  0, 5, 10, 8, 7},
                {0,  5, 0,  5, 3, 6},
                {0, 10, 5,  0, 1, 3},
                {0,  8, 3,  1, 0, 1},
                {0,  7, 6,  3, 1, 0}};


        List<Integer> find = new ArrayList<>();
        List<Integer> value = new ArrayList<>();
        boolean[] visit = new boolean[N+1];
        int sum=0;

        find.add(1);
        value.add(0);

        while(find.size()!=N) {
            int to =0;
            int min=Integer.MAX_VALUE;
            for(int i :find) {
                for(int j=1;j<=N;j++) {
                    if(!visit[j] && ia[i][j]!=0){
                        if(ia[i][j]<min) {
                            min=ia[i][j];
                            to=j;
                        }
                    }
                }
            }
            find.add(to);
            value.add(min);
            visit[to] = true;
            sum+=min;
        }

        System.out.println(find);
        System.out.println(value);
        System.out.println(sum);

    }


}


/*
5
0 5 10 8 7
5 0 5 3 6
10 5 0 1 3
8 3 1 0 1
7 6 3 1 0

 10
 */