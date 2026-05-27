"""
Testes automatizados do pipeline de Odometria Visual.
Cobre: YAML, config, constantes OpenCV, argumentos CLI, detector ORB/AKAZE,
       resolução de device (auto/cpu/cuda), e o bloco de imports neurais.
"""

import sys, os, tempfile, textwrap, subprocess, unittest
sys.path.insert(0, os.path.dirname(__file__))

import numpy as np
import cv2 as cv
import yaml

# Módulos a testar
import main as M
from odometria_visual import (
    OdometriaVisual,
    TORCH_AVAILABLE,
    estima_latlon,
    calcula_distancia_latlon,
    criar_caminho_kml,
)

# ─────────────────────────── helpers ────────────────────────────────────────

MINIMAL_YAML = textwrap.dedent("""\
    paths:
      base_path: "/tmp/vo_test"
      map_tif:      "map.tif"
      images:       "imgs"
      ground_truth: "gt.csv"
      output:       "out"
    camera:
      width: 640
      height: 640
      h_fov: 71.56
      v_fov: 71.56
    detector: ORB
    device: auto
    detector_params:
      orb:
        nfeatures: 500
        scaleFactor: 1.2
        nlevels: 8
        edgeThreshold: 31
        firstLevel: 0
        WTA_K: 2
        scoreType: HARRIS
        patchSize: 31
      akaze:
        descriptor_type: MLDB
        descriptor_size: 0
        descriptor_channels: 3
        threshold: 0.001
        nOctaves: 4
        nOctaveLayers: 4
        diffusivity: PM_G2
    matcher:
      nn_match_ratio: 0.8
    map_matching:
      enabled: false
      interval: 1
      roi_size_m: 1000
    display:
      show_plot: false
      show_images: false
""")

def write_yaml(content=MINIMAL_YAML):
    f = tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False)
    f.write(content)
    f.close()
    return f.name

def make_config(extra=None):
    cfg = yaml.safe_load(MINIMAL_YAML)
    if extra:
        cfg.update(extra)
    return M.montar_config(cfg)

def make_orb_ov(device="auto"):
    cfg = yaml.safe_load(MINIMAL_YAML)
    cfg["device"] = device
    config = M.montar_config(cfg)
    # Substituímos caminhos de dados por None para não precisar de disco
    config["paths"] = {
        "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
        "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
    }
    return OdometriaVisual(config)


# ─────────────────────────── testes ─────────────────────────────────────────

class TestYAMLLoading(unittest.TestCase):
    def test_load_valid_yaml(self):
        path = write_yaml()
        cfg = M.carregar_yaml(path)
        self.assertEqual(cfg["detector"], "ORB")
        os.unlink(path)

    def test_load_missing_file(self):
        with self.assertRaises(FileNotFoundError):
            M.carregar_yaml("/nao/existe.yaml")


class TestResolverCamera(unittest.TestCase):
    def test_fx_fy_positive(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cam = M.resolver_camera(cfg)
        self.assertGreater(cam["fx"], 0)
        self.assertGreater(cam["fy"], 0)
        self.assertAlmostEqual(cam["cx"], 320.0)
        self.assertAlmostEqual(cam["cy"], 320.0)

    def test_square_image_equal_fov(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cam = M.resolver_camera(cfg)
        self.assertAlmostEqual(cam["fx"], cam["fy"])


class TestOpencvConstants(unittest.TestCase):
    def test_orb_harris_score(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        p = M.resolver_detector_params(cfg)
        self.assertEqual(p["orb"]["scoreType"], cv.ORB_HARRIS_SCORE)

    def test_orb_fast_score(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg["detector_params"]["orb"]["scoreType"] = "FAST"
        p = M.resolver_detector_params(cfg)
        self.assertEqual(p["orb"]["scoreType"], cv.ORB_FAST_SCORE)

    def test_akaze_mldb(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        p = M.resolver_detector_params(cfg)
        self.assertEqual(p["akaze"]["descriptor_type"], cv.AKAZE_DESCRIPTOR_MLDB)

    def test_akaze_pm_g2(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        p = M.resolver_detector_params(cfg)
        self.assertEqual(p["akaze"]["diffusivity"], cv.KAZE_DIFF_PM_G2)

    def test_unknown_score_defaults_to_harris(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg["detector_params"]["orb"]["scoreType"] = "INVALID"
        p = M.resolver_detector_params(cfg)
        self.assertEqual(p["orb"]["scoreType"], cv.ORB_HARRIS_SCORE)


class TestMontarConfig(unittest.TestCase):
    def test_device_default_auto(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg.pop("device", None)
        config = M.montar_config(cfg)
        self.assertEqual(config["device"], "auto")

    def test_device_cpu(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg["device"] = "cpu"
        config = M.montar_config(cfg)
        self.assertEqual(config["device"], "cpu")

    def test_detector_type_uppercase(self):
        config = make_config()
        self.assertEqual(config["detector_type"], "ORB")

    def test_paths_built_from_base(self):
        config = make_config()
        self.assertIn("/tmp/vo_test", config["paths"]["image_path"])


class TestArgParser(unittest.TestCase):
    def _parse(self, args):
        old = sys.argv
        sys.argv = ["main.py"] + args
        result = M.parse_args()
        sys.argv = old
        return result

    def test_default_config(self):
        args = self._parse([])
        self.assertEqual(args.config, "config.yaml")

    def test_detector_override(self):
        args = self._parse(["--detector", "AKAZE"])
        self.assertEqual(args.detector, "AKAZE")

    def test_device_cpu(self):
        args = self._parse(["--device", "cpu"])
        self.assertEqual(args.device, "cpu")

    def test_device_auto(self):
        args = self._parse(["--device", "auto"])
        self.assertEqual(args.device, "auto")

    def test_invalid_device_exits(self):
        with self.assertRaises(SystemExit):
            self._parse(["--device", "tpu"])

    def test_invalid_detector_exits(self):
        with self.assertRaises(SystemExit):
            self._parse(["--detector", "SIFT"])


class TestResolverDevice(unittest.TestCase):
    def test_cpu_forced(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch não instalado")
        import torch
        ov = make_orb_ov(device="cpu")
        self.assertEqual(str(ov.device), "cpu")

    def test_auto_falls_back_to_cpu_without_cuda(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch não instalado")
        import torch
        ov = make_orb_ov(device="auto")
        # Sem GPU no ambiente de teste → deve ser cpu
        expected = "cuda" if torch.cuda.is_available() else "cpu"
        self.assertEqual(str(ov.device), expected)

    def test_cuda_raises_without_gpu(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch não instalado")
        import torch
        if torch.cuda.is_available():
            self.skipTest("GPU disponível — teste não se aplica")
        with self.assertRaises(RuntimeError):
            make_orb_ov(device="cuda")


class TestDetectorInit(unittest.TestCase):
    def test_orb_creates_detector(self):
        ov = make_orb_ov()
        self.assertIsNotNone(ov.detector)
        self.assertIsNotNone(ov.matcher)

    def test_akaze_creates_detector(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg["detector"] = "AKAZE"
        config = M.montar_config(cfg)
        config["paths"] = {
            "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
            "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
        }
        ov = OdometriaVisual(config)
        self.assertIsNotNone(ov.detector)

    def test_unknown_detector_raises(self):
        cfg = yaml.safe_load(MINIMAL_YAML)
        cfg["detector"] = "SIFT"
        config = M.montar_config(cfg)
        config["paths"] = {
            "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
            "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
        }
        with self.assertRaises(ValueError):
            OdometriaVisual(config)


class TestSuperPointCPU(unittest.TestCase):
    """
    Testa que SuperPoint+LightGlue são inicializados corretamente na CPU.
    Usa mock para evitar download de pesos de rede (feito apenas na 1a execucao real).
    """

    def setUp(self):
        if not TORCH_AVAILABLE:
            self.skipTest("torch nao instalado")
        lg_path = os.path.join(os.path.dirname(__file__), "LightGlue")
        if not os.path.isdir(lg_path):
            self.skipTest("LightGlue nao clonado")

    def _mock_model(self, captured_device=None):
        from unittest.mock import MagicMock
        m = MagicMock()
        m.eval.return_value = m
        if captured_device is not None:
            m.to.side_effect = lambda d: setattr(captured_device, "v", d) or m
        else:
            m.to.return_value = m
        return m

    def test_superpoint_uses_cpu_device(self):
        """Verifica que SuperPoint e LightGlue sao enviados para device=cpu."""
        import torch
        from unittest.mock import patch
        import odometria_visual as ov_mod

        sp_dev, lg_dev = type("H", (), {})(), type("H", (), {})()

        with patch.object(ov_mod, "SuperPoint", return_value=self._mock_model(sp_dev)), \
             patch.object(ov_mod, "LightGlue",  return_value=self._mock_model(lg_dev)):

            cfg = yaml.safe_load(MINIMAL_YAML)
            cfg["detector"] = "SUPERPOINT"
            cfg["device"]   = "cpu"
            config = M.montar_config(cfg)
            config["paths"] = {
                "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
                "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
            }
            OdometriaVisual(config)

        self.assertEqual(str(sp_dev.v), "cpu")
        self.assertEqual(str(lg_dev.v), "cpu")
        print("  SuperPoint + LightGlue: device=cpu confirmado")

    def test_superpoint_params_passed_correctly(self):
        """Verifica que os parametros do YAML chegam ao construtor do SuperPoint."""
        from unittest.mock import patch, MagicMock
        import odometria_visual as ov_mod

        captured = {}

        def fake_superpoint(**kwargs):
            captured["kwargs"] = kwargs
            m = MagicMock()
            m.eval.return_value = m
            m.to.return_value = m
            return m

        with patch.object(ov_mod, "SuperPoint", side_effect=fake_superpoint), \
             patch.object(ov_mod, "LightGlue",  return_value=self._mock_model()):

            cfg = yaml.safe_load(MINIMAL_YAML)
            cfg["detector"] = "SUPERPOINT"
            cfg["device"]   = "cpu"
            cfg["detector_params"]["superpoint"] = {"max_num_keypoints": 512}
            config = M.montar_config(cfg)
            config["paths"] = {
                "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
                "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
            }
            OdometriaVisual(config)

        self.assertEqual(captured["kwargs"].get("max_num_keypoints"), 512)
        print(f"  SuperPoint params recebidos: {captured['kwargs']}")

    def test_superpoint_auto_device_without_cuda(self):
        """Com device=auto sem GPU, resolve automaticamente para cpu."""
        import torch
        from unittest.mock import patch
        import odometria_visual as ov_mod

        if torch.cuda.is_available():
            self.skipTest("GPU disponivel - nao se aplica")

        with patch.object(ov_mod, "SuperPoint", return_value=self._mock_model()), \
             patch.object(ov_mod, "LightGlue",  return_value=self._mock_model()):

            cfg = yaml.safe_load(MINIMAL_YAML)
            cfg["detector"] = "SUPERPOINT"
            cfg["device"]   = "auto"
            config = M.montar_config(cfg)
            config["paths"] = {
                "image_path": "/tmp", "ground_truth_path": "/tmp/gt.csv",
                "output_dir": "/tmp", "map_tif_path": "/tmp/map.tif",
            }
            ov = OdometriaVisual(config)

        self.assertEqual(str(ov.device), "cpu")
        print("  device=auto sem GPU resolveu para cpu")


class TestGeoHelpers(unittest.TestCase):
    def test_estima_latlon_north(self):
        lat, lon = estima_latlon(-23.0, -46.0, 0.0, 1000.0)
        self.assertGreater(lat, -23.0)   # moveu para norte

    def test_calcula_distancia_zero(self):
        d = calcula_distancia_latlon(-23.0, -46.0, -23.0, -46.0)
        self.assertAlmostEqual(d, 0.0, places=3)

    def test_calcula_distancia_positiva(self):
        d = calcula_distancia_latlon(-23.0, -46.0, -22.0, -46.0)
        self.assertGreater(d, 0)

    def test_kml_file_created(self):
        with tempfile.NamedTemporaryFile(suffix=".kml", delete=False) as f:
            path = f.name
        criar_caminho_kml([-23.0, -23.1], [-46.0, -46.1], path)
        self.assertTrue(os.path.exists(path))
        with open(path) as f:
            content = f.read()
        self.assertIn("<kml", content)
        self.assertIn("-46.0,-23.0,0", content)
        os.unlink(path)


class TestORBCorrespondencias(unittest.TestCase):
    def test_orb_matches_synthetic_images(self):
        ov = make_orb_ov()
        rng = np.random.default_rng(0)
        img1 = (rng.random((128, 128)) * 255).astype(np.uint8)
        img2 = np.roll(img1, 5, axis=1)   # shift horizontal pequeno
        pts1, pts2, kp1, kp2, n = ov._obter_correspondencias(img1, img2)
        print(f"  ORB: {kp1} kp1, {kp2} kp2, {n} matches")
        self.assertIsNotNone(pts1)

    def test_orb_few_features_returns_none(self):
        ov = make_orb_ov()
        blank1 = np.zeros((64, 64), dtype=np.uint8)
        blank2 = np.zeros((64, 64), dtype=np.uint8)
        pts1, pts2, kp1, kp2, n = ov._obter_correspondencias(blank1, blank2)
        # Imagens sem textura → sem descritores → retorna None
        self.assertIsNone(pts1)


class TestCalcDeslocamento(unittest.TestCase):
    def test_scale_estimate(self):
        pts1 = np.array([[0., 0.], [10., 0.]])
        pts2 = np.array([[5., 0.], [15., 0.]])
        d = OdometriaVisual._calcula_deslocamento_escala(pts1, pts2, gsd=0.5)
        self.assertAlmostEqual(d, 2.5)   # 5 pixels * 0.5 m/pixel

    def test_empty_returns_zero(self):
        d = OdometriaVisual._calcula_deslocamento_escala([], [], gsd=1.0)
        self.assertEqual(d, 0.0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
