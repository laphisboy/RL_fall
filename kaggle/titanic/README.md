## Reference

Attempt based on   

https://github.com/minsuk-heo/kaggle-titanic

I made the results worse...

#### Trial 2

little improvement from 0.76794 --> 0.77033

#### Conclusion

probably ignoring missing data,  
or giving it some other value on personal whim  
is probably not a good idea  

Rather like the solution file referenced to,  
filling in missing data with more probable 'median',
and making age groups will have better results

#### Ideas on further improvement

instead of filling in missing data with median,  
it could be filled in by generating a gaussian distribution according the mean and variance  
that can be calculated with the data present
