package com.insam.JavaAlgo;


import java.util.Arrays;
import java.util.Scanner;

public class devideConquer {

    //http://www.jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=614&sca=3060

    static int blue=0;
    static int white=0;
    static int N=0;
    public static void main(String arsg[]) {
  /*
  System.out.println("Abc");
  Scanner s = new Scanner(System.in);
  N = s.nextInt();
  int[][] pa = new int[N][N];
  for(int i=0; i<N; i++) {
   for(int j=0; j<N; j++) {
    pa[i][j]=s.nextInt();
   }

  }
  */
        N = 8;
        int[][] pa = {
                {1 ,1 ,0 ,0 ,0 ,0 ,1 ,1},
                {1 ,1 ,0 ,0 ,0 ,0 ,1 ,1},
                {0 ,0 ,0 ,0 ,1 ,1 ,0 ,0},
                {0 ,0 ,0 ,0 ,1 ,1 ,0 ,0},
                {1 ,0 ,0 ,0 ,1 ,1 ,1 ,1},
                {0 ,1 ,0 ,0 ,1 ,1 ,1 ,1},
                {0 ,0 ,1 ,1 ,1 ,1 ,1 ,1},
                {0 ,0 ,1 ,1 ,1 ,1 ,1 ,1},
        };

        for(int[] a : pa) System.out.println(Arrays.toString(a));

        count(pa,0,0,N);

    }

    public static void count(int[][] pa,int r, int c,int n) {
        int v=pa[r][c];
        boolean flag=true;

        EXIT:
        for(int i=r; i<r+n; i++) {
            for(int j=c; j<c+n; j++) {
                if(v!=pa[i][j]) {
                    flag=false;
                    //break; // 해당 for 문만 빠져나감
                    break EXIT; // 하나라도 색이 틀리면 빠져 나가자
                }
            }
        }

        if(flag) {
            if(v==1) blue++;
            else white++;
            return;
        } else {
            //1,2
            //3,4
            //1 좌상
            count(pa,r,c,n/2);
            //2 우상
            count(pa,r,c+n/2,n/2);
            //3 좌하
            count(pa,r+n/2,c,n/2);
            //4 우하
            count(pa,c+n/2,r+n/2,n/2);
        }
    }
}

/*
8
1 1 0 0 0 0 1 1
1 1 0 0 0 0 1 1
0 0 0 0 1 1 0 0
0 0 0 0 1 1 0 0
1 0 0 0 1 1 1 1
0 1 0 0 1 1 1 1
0 0 1 1 1 1 1 1
0 0 1 1 1 1 1 1

9
7
*/