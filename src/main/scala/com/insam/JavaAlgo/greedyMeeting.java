package com.insam.JavaAlgo;

import java.util.ArrayList;
import java.util.Arrays;
import java.util.List;

class Meeting implements Comparable<Meeting>{
    private int num;
    private int start;
    private int end;
    public Meeting() {

    };
    public Meeting(int num, int start, int end) {
        this.num=num;
        this.start=start;
        this.end=end;
    }
    public int getNum() {
        return num;
    }
    public void setNum(int num) {
        this.num = num;
    }
    public int getStart() {
        return start;
    }
    public void setStart(int start) {
        this.start = start;
    }
    public int getEnd() {
        return end;
    }
    public void setEnd(int end) {
        this.end = end;
    }
    @Override
    public int compareTo(Meeting o) {
        // TODO Auto-generated method stub
        int t= end - o.getEnd();
        if(t==0)
            t=start - o.getStart();
        return t;
    };

    @Override
    public String toString() {
        return num + ", " + start + ", " + end;
    }

}



public class greedyMeeting {
    // 회의가 겹치지 않게 하면서 회의실을 사용할 수 있는 최대의 회의수와 그때의 회의들의 순서를 출력하는 프로그램 작성

    public static Meeting[] meetAll = new Meeting[9];
    public static List<Meeting> sche = new ArrayList<Meeting>();
    public static List<Meeting> sche_result = new ArrayList<Meeting>();

    public static void main(String args[]) {
        System.out.println("GreedyMeeting");

        //Meeting meet = new Meeting(1,3,5);
        meetAll[0] = new Meeting(1,3,5);
        meetAll[1] = new Meeting(2,1,4);
        meetAll[2] = new Meeting(3,2,13);
        meetAll[3] = new Meeting(4,5,9);
        meetAll[4] = new Meeting(5,5,7);
        meetAll[5] = new Meeting(6,0,6);
        meetAll[6] = new Meeting(7,8,11);
        meetAll[7] = new Meeting(8,8,12);
        meetAll[8] = new Meeting(9,12,14);

        //System.out.println(Arrays.toString(meetAll));

        sche_result = getSchedule();
        for(Meeting meet :sche_result) {
            System.out.println(meet.toString());
        }


    }

    public static List<Meeting> getSchedule(){
        Arrays.sort(meetAll);
        sche.add(meetAll[0]);
        int count=1;
        for(int j = 1; j < meetAll.length; j++) {
            if(sche.get(count-1).getEnd() <= meetAll[j].getStart()) {
                sche.add(meetAll[j]);
                count++;
            }
        }

        return sche;

    }
}



//http://www.jungol.co.kr/bbs/board.php?bo_table=pbank&wr_id=645&sca=3020
/*
회의실이 하나 있다. 여러 회의들이 시작시간과 종료시간이 예약되어 있으며, 시간대가 겹치는 회의는 동시에 개최가 불가능하다. 따라서 같은 시간대에 속하는 회의들 중 하나만 개최하고 나머지 회의들은 버려야한다.

단, 종료시간과 시작시간이 같은 경우에는 시간이 겹친다고 말하지 않는다. 회의의 개수 N과 각 회의의 시작시간, 종료시간이 주어졌을 때 되도록 많은 회의를 개최하고자 한다.



회의를 최대한 많이 배정하는 프로그램을 작성하시오.


첫줄에는 회의의 수 N(5≤N≤500), 둘째 줄부터 i-1번 회의의 번호와 시작시간과 종료시간이 차례로 주어진다. (500 이하의 자연수)



첫줄에는 배정 가능한 최대의 회의수를 출력하고 다음 줄부터는 배정한 회의의 번호를 시간대순으로 출력한다. 만약, 답이 여러 가지(최대회의수가 될 수 있는 배정 방법이 여러가지)라면 그 중 아무거나 하나 출력한다.
 */

