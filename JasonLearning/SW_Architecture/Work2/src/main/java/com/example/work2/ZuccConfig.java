package com.example.work2;

import org.springframework.boot.context.properties.ConfigurationProperties;
import org.springframework.boot.context.properties.EnableConfigurationProperties;
import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.context.annotation.PropertySource;

@Configuration
@EnableConfigurationProperties(ZuccConfig.class)
@PropertySource("classpath:zucc.properties")
@ConfigurationProperties(prefix = "zucc")
public class ZuccConfig {
    private int year;

    @Bean
    public OurSchool ourSchool() {
        System.out.println(year);
        if (year < 2020) {
            return new OurSchool("浙江大学城市学院");
        } else {
            return new OurSchool("浙大城市学院");
        }
    }

    public int getYear() {
        return year;
    }

    public void setYear(int year) {
        this.year = year;
    }

}
