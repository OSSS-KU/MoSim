/// GCD of two numbers.
fn gcd(mut a: i64, mut b: i64) -> i64 {
    a = a.abs();
    b = b.abs();
    while b != 0 {
        let t = b;
        b = a % b;
        a = t;
    }
    a
}

/// LCM of two numbers.
fn lcm_pair(a: i64, b: i64) -> i64 {
    if a == 0 || b == 0 {
        return 0;
    }
    (a / gcd(a, b)) * b
}

/// Compute LCM of a list of numbers using gcd_fold algorithm.
pub fn lcm_many(numbers: &[i64]) -> i64 {
    if numbers.is_empty() {
        panic!("Cannot compute LCM of empty list");
    }
    if numbers.contains(&0) {
        return 0;
    }
    let abs_numbers: Vec<i64> = numbers.iter().map(|n| n.abs()).collect();
    if abs_numbers.len() == 1 {
        return abs_numbers[0];
    }
    abs_numbers.iter().copied().reduce(lcm_pair).unwrap()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_gcd_basic() {
        assert_eq!(gcd(12, 8), 4);
        assert_eq!(gcd(100, 75), 25);
        assert_eq!(gcd(7, 13), 1);
    }

    #[test]
    fn test_lcm_pair() {
        assert_eq!(lcm_pair(4, 6), 12);
        assert_eq!(lcm_pair(3, 7), 21);
    }

    #[test]
    fn test_lcm_many() {
        assert_eq!(lcm_many(&[4, 6, 10]), 60);
        assert_eq!(lcm_many(&[2, 3, 5, 7]), 210);
        assert_eq!(lcm_many(&[12]), 12);
    }

    #[test]
    fn test_lcm_with_zero() {
        assert_eq!(lcm_many(&[0, 5, 10]), 0);
    }

    #[test]
    #[should_panic]
    fn test_lcm_empty() {
        lcm_many(&[]);
    }
}
