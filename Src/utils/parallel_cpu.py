


def divide_list(lst, n):
   
    base_length = len(lst) // n
    
    remainder = len(lst) % n
  
    result = []


    start = 0
    for i in range(n):
     
        end = start + base_length + (1 if i < remainder else 0)
    
        result.append(lst[start:end])
     
        start = end

    return result