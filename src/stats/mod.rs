// The code in this module has been adapted from
// the [statrs](https://github.com/statrs-dev/statrs?tab=MIT-1-ov-file) package

// MIT License
//
// Copyright (c) 2016 Michael Ma
//
// Permission is hereby granted, free of charge, to any person obtaining a copy of this software
// and associated documentation files (the "Software"), to deal in the Software without
// restriction, including without limitation the rights to use, copy, modify, merge, publish,
// distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the
// Software is furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in all copies or
// substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING
// BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND
// NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM,
// DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

#[allow(warnings)]
mod erf;
#[allow(warnings)]
mod evaluate;

pub(crate) struct Normal {
    mean: f64,
    std_dev: f64,
}

impl Default for Normal {
    fn default() -> Self {
        Self {
            mean: 0.0,
            std_dev: 1.0,
        }
    }
}

impl Normal {
    pub fn cdf(&self, x: f64) -> f64 {
        cdf_unchecked(x, self.mean, self.std_dev)
    }
}

pub fn cdf_unchecked(x: f64, mean: f64, std_dev: f64) -> f64 {
    0.5 * erf::erfc((mean - x) / (std_dev * std::f64::consts::SQRT_2))
}
