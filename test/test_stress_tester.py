import unittest
from unittest.mock import patch, MagicMock
import time
import psutil
import torch
from teleai_pipe.tool.stress_tester import (
    Sample,
    Report,
    RuntimeMonitor,
    measure_performance_with_monitor,
)


class TestRuntimeMonitor(unittest.TestCase):
    def setUp(self):
        self.monitor = RuntimeMonitor(interval=0.1)

    def test_start_stop(self):
        # 测试开始和停止监控
        self.assertFalse(self.monitor.monitoring)
        self.monitor.start()
        self.assertTrue(self.monitor.monitoring)
        self.monitor.stop()
        self.assertFalse(self.monitor.monitoring)

    def test_get_samples(self):
        # 测试获取采样数据
        self.monitor.samples = [Sample("2023-10-01 12:00:00", 50.0, 100.0, 0.0)]
        samples = self.monitor.get_samples()
        self.assertEqual(len(samples), 1)
        self.assertEqual(samples[0].timestamp, "2023-10-01 12:00:00")
        self.assertEqual(samples[0].cpu_usage, 50.0)
        self.assertEqual(samples[0].memory_usage, 100.0)
        self.assertEqual(samples[0].gpu_memory, 0.0)

    def test_clear_samples(self):
        # 测试清除历史采样数据
        self.monitor.samples = [Sample("2023-10-01 12:00:00", 50.0, 100.0, 0.0)]
        self.monitor.clear_samples()
        self.assertEqual(len(self.monitor.samples), 0)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.Process")
    @patch("time.time")
    def test_monitor_loop(
        self, mock_time, mock_process, mock_virtual_memory, mock_cpu_percent
    ):
        # 模拟时间戳
        mock_time.side_effect = [1696142400.0, 1696142400.1, 1696142400.2]

        # 模拟CPU使用率
        mock_cpu_percent.return_value = 50.0

        # 模拟内存使用情况
        mock_virtual_memory.return_value = psutil._pslinux.svmem(
            total=1000000000,
            available=500000000,
            percent=50.0,
            used=500000000,
            free=500000000,
            active=250000000,
            inactive=250000000,
            buffers=100000000,
            cached=200000000,
            shared=150000000,
            slab=100000000,
        )

        # 模拟进程信息
        mock_process.return_value.cpu_percent.return_value = 50.0
        mock_process.return_value.memory_info.return_value = psutil._pslinux.pmem(
            rss=100000000,
            vms=200000000,
            shared=300000000,
            text=400000000,
            lib=500000000,
            data=600000000,
            dirty=700000000,
        )

        # 开始监控
        self.monitor.start()
        time.sleep(0.3)  # 等待一段时间以确保监控循环执行
        self.monitor.stop()

        # 验证采样数据
        self.assertGreater(len(self.monitor.samples), 0)

    @patch("psutil.cpu_percent")
    @patch("psutil.virtual_memory")
    @patch("psutil.Process")
    @patch("time.time")
    def test_measure_performance_with_monitor(
        self, mock_time, mock_process, mock_virtual_memory, mock_cpu_percent
    ):
        # 模拟时间戳
        mock_time.side_effect = [1696142400.0, 1696142400.1, 1696142400.2, 1696142400.3]

        # 模拟CPU使用率
        mock_cpu_percent.side_effect = [50.0, 60.0]

        # 模拟内存使用情况
        mock_virtual_memory.side_effect = [
            psutil._pslinux.svmem(
                total=1000000000,
                available=500000000,
                percent=50.0,
                used=500000000,
                free=500000000,
                active=250000000,
                inactive=250000000,
                buffers=100000000,
                cached=200000000,
                shared=150000000,
                slab=100000000,
            ),
            psutil._pslinux.svmem(
                total=1000000000,
                available=400000000,
                percent=60.0,
                used=600000000,
                free=400000000,
                active=300000000,
                inactive=300000000,
                buffers=120000000,
                cached=240000000,
                shared=180000000,
                slab=120000000,
            ),
        ]

        # 模拟进程信息
        mock_process.return_value.cpu_percent.side_effect = [50.0, 60.0]
        mock_process.return_value.memory_info.side_effect = [
            psutil._pslinux.pmem(
                rss=100000000,
                vms=200000000,
                shared=300000000,
                text=400000000,
                lib=500000000,
                data=600000000,
                dirty=700000000,
            ),
            psutil._pslinux.pmem(
                rss=120000000,
                vms=240000000,
                shared=360000000,
                text=480000000,
                lib=600000000,
                data=720000000,
                dirty=840000000,
            ),
        ]

        # 定义一个简单的测试函数
        def test_func():
            time.sleep(0.2)
            return "Test Result"

        # 执行性能测量
        report = measure_performance_with_monitor(test_func)

        # 验证报告内容
        self.assertEqual(report.function_name, "test_func")
        self.assertAlmostEqual(report.execution_time, 0.2, delta=0.1)
        self.assertAlmostEqual(report.initial_cpu, 50.0, delta=0.1)
        self.assertAlmostEqual(report.final_cpu, 60.0, delta=0.1)
        self.assertEqual(report.monitor_interval, 0.1)
        self.assertGreater(len(report.samples), 0)


if __name__ == "__main__":
    unittest.main()
