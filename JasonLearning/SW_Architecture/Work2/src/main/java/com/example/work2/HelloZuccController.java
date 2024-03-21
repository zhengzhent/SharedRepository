package com.example.work2;

import org.springframework.beans.factory.annotation.Autowired;
import org.springframework.web.bind.annotation.RequestMapping;
import org.springframework.web.bind.annotation.RestController;

@RestController
public class HelloZuccController {
    @Autowired
    private OurSchool ourSchool;

    @RequestMapping("/school")
    public String hello(){
        return ourSchool.getName();
    }
}
