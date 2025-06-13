import numpy as np
from scipy import integrate
import matplotlib.pyplot as plt
import logging

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class TunnelSettlement:
    def __init__(self, H, R, Vl, u1, u3, theta, fai, k):
        """
        初始化隧道沉降计算类
        
        参数:
            H: float, 隧道埋深
            R: float, 隧道半径
            Vl: float, 地层损失率
            u1: float, 第一偏压参数
            u3: float, 第三偏压参数
            theta: float, 偏压角度（弧度）
            fai: float, 平均内摩擦角
            k: float, 土质参数
        """
        self.H = H
        self.R = R
        self.Vl = Vl
        self.u1 = u1
        self.u3 = u3
        self.theta = theta
        self.fai = fai
        self.k = k
        
        # 计算中间参数
        self.u0 = R - R * np.sqrt(1 - Vl)
        self.u2 = R * np.sqrt(1 - Vl) - R * (1 - Vl) / (np.sqrt(1 - Vl) + u1/R)
        self.Z = H - R
        self.A = H - R
        self.B = H + R
        self.tanb = H / (np.sqrt(2 * np.pi) * (1 - 0.02 * fai) * (H - k * self.A))
        
        # 计算积分区间
        self.A1 = -R + self.u0 + u1
        self.B1 = -self.A1

    def _calculate_C_D(self, mu):
        """计算C和D值"""
        C = -np.sqrt(self.R**2 - (mu - self.H)**2)
        return C, -C

    def _calculate_C1_D1(self, muhat):
        """计算C1和D1值"""
        C1 = -(self.R - self.u0 + self.u2) * np.sqrt(1 - (muhat/(self.R - self.u0 - self.u1))**2)
        return C1, -C1

    def f_function(self, x, mu, sigma):
        """计算f(x)函数"""
        f1 = self.tanb / mu
        f2 = -np.pi * self.tanb**2 * (x - sigma)**2 / mu**2
        return f1 * np.exp(f2)

    def g_function(self, x, muhat, sigma_hat):
        """计算g(x)函数"""
        g11 = self.H + (self.u3 + muhat) * np.cos(self.theta) + sigma_hat * np.sin(self.theta)
        g1 = self.tanb / g11
        g21 = -np.pi * self.tanb**2 * (x + (self.u3 + muhat) * np.sin(self.theta) - sigma_hat * np.cos(self.theta))**2
        g22 = g11**2
        g2 = g21 / g22
        return g1 * np.exp(g2)

    def F_integrand(self, sigma, mu, x):
        """F(x)的内层被积函数"""
        return self.f_function(x, mu, sigma)

    def G_integrand(self, sigma_hat, muhat, x):
        """G(x)的内层被积函数"""
        return self.g_function(x, muhat, sigma_hat)

    def calculate_F(self, x):
        """计算F(x)"""
        def outer_integrand(mu):
            C, D = self._calculate_C_D(mu)
            inner_integral, _ = integrate.quad(lambda sigma: self.F_integrand(sigma, mu, x), C, D)
            return inner_integral
        
        F, _ = integrate.quad(outer_integrand, self.A, self.B)
        return F

    def calculate_G(self, x):
        """计算G(x)"""
        def outer_integrand(muhat):
            C1, D1 = self._calculate_C1_D1(muhat)
            inner_integral, _ = integrate.quad(lambda sigma_hat: self.G_integrand(sigma_hat, muhat, x), C1, D1)
            return inner_integral
        
        G, _ = integrate.quad(outer_integrand, self.A1, self.B1)
        return G

    def calculate_W(self, x):
        """计算W(x) = F(x) - G(x)"""
        return self.calculate_F(x) - self.calculate_G(x)

def get_user_input():
    """获取用户输入的参数"""
    print("\n请输入计算参数：")
    H = float(input("隧道埋深 (H): "))
    R = float(input("隧道半径 (R): "))
    Vl = float(input("地层损失率 (Vl): "))
    u1 = float(input("第一偏压参数 (u1): "))
    u3 = float(input("第三偏压参数 (u3): "))
    theta = float(input("偏压角度 (theta，度): ")) * np.pi / 180  # 转换为弧度
    fai = float(input("平均内摩擦角 (fai): "))
    k = float(input("土质参数 (k): "))
    return H, R, Vl, u1, u3, theta, fai, k

def main():
    try:
        # 获取用户输入
        H, R, Vl, u1, u3, theta, fai, k = get_user_input()
        
        # 创建计算实例
        calculator = TunnelSettlement(H, R, Vl, u1, u3, theta, fai, k)
        
        # 生成x值范围
        x_range = np.linspace(-3*R, 3*R, 100)
        
        # 计算W(x)值
        W_values = [calculator.calculate_W(x) for x in x_range]
        
        # 绘制图形
        plt.figure(figsize=(10, 6))
        plt.plot(x_range, W_values, 'b-', label='W(x)')
        plt.xlabel('x')
        plt.ylabel('W(x)')
        plt.title('地表沉降曲线')
        plt.grid(True)
        plt.legend()
        
        # 保存图形
        plt.savefig('settlement_curve.png')
        plt.close()
        
        logger.info("计算完成，图形已保存为 'settlement_curve.png'")
        
    except Exception as e:
        logger.error(f"计算过程中出现错误: {str(e)}")
        raise

if __name__ == "__main__":
    main() 