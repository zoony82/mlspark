package com.insam.JavaAlgo;

import java.util.Arrays;
import java.util.LinkedList;
import java.util.Queue;
import java.util.Scanner;

public class dfsBfs {
    public static int N;
    public static int[][] ia=null;
    public static boolean[] visit = null;
    public static StringBuilder sb = null;


    public static void main(String[] args) {

        //https://www.acmicpc.net/problem/1260

        // TODO Auto-generated method stub
        Scanner s = new Scanner(System.in);
        N = s.nextInt();
        int M = s.nextInt();
        int S = s.nextInt();
        //System.out.println(N + " " + M + " " + S);

        System.out.println("\n============\n");
        System.out.println("============\n");


        ia = new int[N+1][N+1];
        for(int i=1; i<=M; i++) {
            //System.out.println(s.nextInt()+ "," + s.nextInt());
            int a=s.nextInt();
            int b=s.nextInt();
            ia[a][b]=1;
            ia[b][a]=1;
        }


        System.out.println(N + " " + M + " " + S);
        for(int[] a:ia) {
            System.out.println(Arrays.toString(a));
        };
//
//  for(int i=1; i<=N; i++) {
//   for(int j=1; j<=N; j++) {
//    System.out.print(ia[i][j] + " ");
//   }
//   System.out.println();
//  }
//
//  for(int[] a:ia)System.out.println(Arrays.toString(a));

        visit = new boolean[N+1];
        System.out.println(Arrays.toString(visit));
        sb = new StringBuilder();
        dfs(S);
        System.out.println(sb.toString());


        visit = new boolean[N+1];
        System.out.println(Arrays.toString(visit));
        sb = new StringBuilder();
        bfs(S);
        System.out.println(sb.toString());
    }

    public static void dfs(int start) {
        sb.append(start + " ");
        visit[start]=true;
        for(int j=1;j<=N;j++) {
            if(start!=j && !visit[j] && ia[start][j]==1) {
                dfs(j);
            }
        }
    }

    public static void bfs(int start) {
        sb.append(start + " ");
        Queue<Integer> queue = new LinkedList<>();
        queue.offer(start);
        visit[start] = true;
        while(!queue.isEmpty()) {
            int i = queue.poll();
            for(int j=1; j<=N; j++) {
                if(i!=j && !visit[j] && ia[i][j] == 1) {
                    queue.offer(j);
                    sb.append(j + " ");
                    visit[j]=true;
                }
            }
        }
    }

}

/*
4 5 1
1 2
1 3
1 4
2 4
3 4

dfs result
1 2 4 3
bfs result
1 2 3 4

*/