package com.insam.JavaAlgo;

import java.util.Arrays;
import java.util.Comparator;


class Employee implements Comparable<Employee>{
    String name;
    int no;
    int salary;
    public Employee(String name, int no, int salary) {
        this.name=name;
        this.no=no;
        this.salary=salary;
    }

    @Override // 암튼 이게 정렬하게 위한거네.. 신기
    public int compareTo(Employee o){

        // 이름으로 정렬
        return name.compareTo(o.name);

    }

    @Override
    public String toString(){
        return name+", "+ no + ", "+salary;
    }
}

// 이름 말고 다른것으로 정렬하고 싶으면,,, 기존 클래스를 수정할 수 없기 때문에 클래스를 하나 더 만들어라
// 비교자를 하나 더 만들어라
class EmpNoComparator implements Comparator<Employee>{
    @Override
    public int compare(Employee o1, Employee o2){
        return o1.no-o2.no;
        // 역순으로 : return o2.no - o1.no;

    }
}

class SalaryComparator implements Comparator<Employee>{
    @Override
    public int compare(Employee o1, Employee o2){


        int d = o1.salary-o2.salary;
        if(d==0){
            return o2.no - o1.no; //샐러리가 같으면 사번으로 역순으로 정렬
        }
        return d;


    }
}


public class employeeSort {
    public static void main(String[] args){
        System.out.println("Test");

        Employee[] ea = {
                new Employee("홍길동",10,3),
                new Employee("임꺽정",150,2),
                new Employee("장준희",300,1)
        };

        //정렬을 하기 위해선 Comparable 인터페이스를 구현 해야 한다.
        Arrays.sort(ea);
        System.out.println(Arrays.toString(ea));

        Arrays.sort(ea,Comparator.reverseOrder());
        System.out.println(Arrays.toString(ea));

        Arrays.sort(ea, new EmpNoComparator());
        System.out.println(Arrays.toString(ea));

        Arrays.sort(ea, new SalaryComparator());
        System.out.println(Arrays.toString(ea));

        // 그런데 클래스 이름 없이도 셋팅 가능하다. 그럴려면 선언을 new 에 해라
        // Anomous Class
        Arrays.sort(ea, new Comparator<Employee>(){
            @Override
            public int compare(Employee o1, Employee o2){
                return o1.no-o2.no;
            }
        });
        System.out.println(Arrays.toString(ea));

        // Convert Lamda
        Arrays.sort(ea, (Employee o1, Employee o2) -> {
                return o1.no-o2.no;
        });
        System.out.println(Arrays.toString(ea));

        Arrays.sort(ea, (o1, o2) -> o1.no-o2.no);
        System.out.println(Arrays.toString(ea));
    }


}
