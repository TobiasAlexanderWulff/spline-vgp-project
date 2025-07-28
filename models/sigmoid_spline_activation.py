import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import grad

import matplotlib.pyplot as plt


class OptimizedSigmoidSpline(nn.Module):
    """Optimized Sigmoid Spline Class"""
    
    def __init__(self, n: int=2, x_limit: int=2):        
        super(OptimizedSigmoidSpline, self).__init__()

        self.n = n
        self.x_limit = x_limit
        
        # Alle Tensoren vollständig von Autograd trennen
        with torch.no_grad():
            self.sig_x = torch.linspace(-self.x_limit, self.x_limit, self.n+1)
            self.sig_y = F.sigmoid(self.sig_x)
            
        # Für die Gradientenberechnung temporäre Tensoren mit requires_grad verwenden
        temp_x = self.sig_x.clone().requires_grad_(True)
        temp_y = F.sigmoid(temp_x)
        self.sig_ys = grad(temp_y, temp_x, torch.ones_like(temp_x))[0]
        
        # Skalarwerte extrahieren
        self._y0s = (self.sig_ys[0] * (self.sig_x[1] - self.sig_x[0])).item()
        self._yns = (self.sig_ys[-1] * (self.sig_x[-1] - self.sig_x[-2])).item()
        self._ys = self._solve_linear_system()


    def _solve_linear_system(self) -> torch.Tensor:
        """
        Solves the linear system to find the coefficients of the spline.
        The system is derived from the Hermite interpolation conditions.
        """
        
        # Matrix A mit eingebauten Randbedingungen:
        # [4, 1, 0, 0, ..., 0]
        # [1, 4, 1, 0, ..., 0]
        # [0, 1, 4, 1, ..., 0]
        # ...
        # [0, 0, ..., 0, 1, 4]
        A = (
            4 * torch.eye(self.n-1) 
            + torch.diag(torch.ones(self.n-2), 1) 
            + torch.diag(torch.ones(self.n-2), -1)
            )
        
        # Vektor b mit eingebauten Randbedingungen:
        # [3 * (y_1 - y_0)]
        # [3 * (y_2 - y_0)]
        # ...
        # [3 * (y_n - y_{n-2})]
        b = 3 * (self.sig_y[2:] - self.sig_y[:-2])
        b[0] -= self._y0s
        b[self.n-2] -= self._yns
        
        # Löse das lineare Gleichungssystem
        c = torch.linalg.solve(A, b)
        
        
        # Füge die Randbedingungen wieder ein
        return torch.cat([torch.tensor([self._y0s]), c, torch.tensor([self._yns])])


    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """
        Forwards pass of the spline function.
        Args:
            t (torch.Tensor): Input tensor.
        Returns:
            torch.Tensor: Spline value.
        """
        return torch.where(
            t < -self.x_limit,
            t + self.sig_y[0].item() + self.x_limit,
            torch.where(
                t > self.x_limit,
                t + self.sig_y[-1].item() - self.x_limit,
                self._spline(t)
            )
        )
    
    
    def _spline(self, t: torch.Tensor) -> torch.Tensor:
        """
        Computes the spline value for t using the precomputed coefficients.
        
        Args:
            t (torch.Tensor): Input tensor.
            
        Returns:
            torch.Tensor: Spline value.
        """
        s = torch.zeros_like(t)
        
        for i in range(self.n):
            s = torch.where(
                (self.sig_x[i] <= t) & (t <= self.sig_x[i+1]),
                self._hermite(self._reparametrize(t, i), self.sig_y[i], self.sig_y[i+1], self._ys[i], self._ys[i+1]),
                s
            )
        return s
    
    
    def _reparametrize(self, t: torch.Tensor, i: int) -> torch.Tensor:
        """
        Reparametrizes the input tensor t to the range of the spline segment.
        
        Args:
            t (torch.Tensor): Input tensor.
            i (int): Index of the spline segment.
            
        Returns:
            torch.Tensor: Reparametrized tensor.
        """
        return (t - self.sig_x[i]) / (self.sig_x[i+1] - self.sig_x[i])
    
    
    def _hermite(self, t: torch.Tensor, y0: float, y1: float, y0s: float, y1s: float) -> torch.Tensor:
        """
        Computes the Hermite polynomial for the given parameters.
        
        Args:
            t (torch.Tensor): Input tensor.
            y0 (float): Value at the start of the segment.
            y1 (float): Value at the end of the segment.
            y0s (float): Slope at the start of the segment.
            y1s (float): Slope at the end of the segment.
            
        Returns:
            torch.Tensor: Hermite polynomial value.
        """
        term1 = torch.pow(t, 3) * (2*y0 - 2*y1 + y0s + y1s)
        term2 = torch.pow(t, 2) * (-3*y0 + 3*y1 - 2*y0s - y1s)
        term3 = t * y0s
        term4 = y0
        return term1 + term2 + term3 + term4
    
    
    def plot(self, t: torch.Tensor):
        """
        Plots the spline function, the original sigmoid function, and the control points.
        Also plots the derivatives accordingly.
        The plot is generated using matplotlib.
        
        Args:
            t (torch.Tensor): Input tensor for plotting.
        """
        plt.figure(figsize=(10, 5))
        
        plt.subplot(1, 2, 1)
        plt.plot(t.cpu().detach().numpy(), F.sigmoid(t).cpu().detach().numpy(), label='Sigmoid', color='green')
        plt.plot(t.cpu().detach().numpy(), self.forward(t).cpu().detach().numpy(), label='Spline', color='blue')
        plt.plot(self.sig_x.detach().numpy(), self.sig_y.detach().numpy(), 'ro', label='Control Points')
        plt.title('Optimized Sigmoid Spline')
        plt.xlabel('t')
        plt.ylabel('y', rotation=0)
        plt.xlim(-4, 4)
        plt.ylim(-0.5, 1.5)
        plt.legend()
        plt.grid()
        
        
        plt.subplot(1, 2, 2)
        plt.plot(t.cpu().detach().numpy(), grad(F.sigmoid(t), t, torch.ones_like(t))[0].cpu().detach().numpy(), label='Sigmoid Derivative', color='green')
        plt.plot(t.cpu().detach().numpy(), grad(self.forward(t), t, torch.ones_like(t), retain_graph=True)[0].cpu().detach().numpy(), label='Spline Derivative', color='orange')
        plt.plot(self.sig_x.detach().numpy(), self.sig_ys.detach().numpy(), 'ro', label='Control Points Derivative')
        plt.title('Optimized Sigmoid Spline Derivative')
        plt.xlabel('t')
        plt.ylabel('dy/dt', rotation=0)
        plt.xlim(-4, 4)
        plt.ylim(-0.5, 2)
        plt.legend()
        plt.grid()
        
        
        plt.tight_layout()
        plt.savefig("spline_vs_sigmoid.png", dpi=300)
        plt.show()
    
    
    def __repr__(self) -> str:
        return f"OptimizedSigmoidSpline(n={self.n}, x_limit={self.x_limit})"
    
    
    def __str__(self) -> str:
        return f"OptimizedSigmoidSpline(n={self.n}, x_limit={self.x_limit})"


if __name__ == "__main__":
    model = OptimizedSigmoidSpline()
    t = torch.linspace(-15, 15, 1000, requires_grad=True, device="cuda")
    y = model(t)
    
    model.plot(t)

