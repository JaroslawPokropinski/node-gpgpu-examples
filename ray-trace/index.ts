import { renderScene, saveToFile } from './Renderer';
import { black, Material, Scene, white } from './Scene';

async function main() {
  const camera = {
    position: { x: 0.0, y: 0.3, z: 1.5 },
    up: { x: 0, y: 1, z: 0 },
    forward: { x: 0, y: 0, z: -1 },
    width: 300,
    height: 300,
  };
  const scene = new Scene(camera.width, camera.height, camera);

  const rstone: Material = { dr: { r: 1, g: 0.5, b: 0 }, sr: black, st: black, e: black };
  const bstone: Material = { dr: { r: 0.01, g: 0.5, b: 1 }, sr: black, st: black, e: black };
  const emmiter: Material = { dr: black, sr: black, st: black, e: { r: 8, g: 8, b: 4 } };

  scene.addSphere(rstone, { x: -0.8, y: 0.5, z: 0.0 }, 0.2);
  scene.addSphere(emmiter, { x: 0.0, y: 0.2, z: 0.4 }, 0.02);
  scene.addSphere(bstone, { x: 0.8, y: 0.5, z: 0.0 }, 0.2);

  // white floor
  scene.addObject(
    { dr: white, sr: black, st: black, e: black },
    { a: { x: -4.0, y: 0, z: 2 }, b: { x: 4, y: 0, z: 2 }, c: { x: 0, y: 0, z: -4 } },
  );

  const render = await renderScene(scene, 1000);
  saveToFile(render, 'image.png', scene.width, scene.height);
}

main();
