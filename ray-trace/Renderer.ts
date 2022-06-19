import { Gpgpu, KernelContext, kernelEntry, kernelFunction, Types } from 'node-gpgpu';
import { createCanvas } from 'canvas';
import { writeFileSync } from 'fs';
import { Camera, Color, Material, Scene, Sphere, TNodes, Triangle, Vector3 } from './Scene';

export type Task = {
  ray: {
    origin: Vector3;
    direction: Vector3;
  };
  weights: Material;
  x: number;
  y: number;
  depth: number;
};

type Ray = {
  origin: Vector3;
  direction: Vector3;
};

export async function renderScene(scene: Scene, iterations: number): Promise<Float32Array> {
  const gpgpu = new Gpgpu();

  const vectorShape = { x: Types.number, y: Types.number, z: Types.number };
  const rayShape: Ray = { origin: vectorShape, direction: vectorShape };
  const colorShape: Color = { r: Types.number, g: Types.number, b: Types.number };
  const materialShape: Material = { dr: colorShape, sr: colorShape, st: colorShape, e: colorShape };
  const tnodesShape: TNodes = { a: vectorShape, b: vectorShape, c: vectorShape };
  const triangleShape: Triangle = {
    d: Types.number,
    materialIdx: Types.number,
    normal: vectorShape,
    position: tnodesShape,
  };
  const sphereShape = {
    position: vectorShape,
    materialIdx: Types.number,
    radius: Types.number,
  };
  const lengthShape = { length: Types.number };
  const infoShape = {
    threads: Types.number,
    iterations: Types.number,
    imageWidth: Types.number,
    imageHeight: Types.number,
  };
  const cameraShape: Camera = {
    position: vectorShape,
    forward: vectorShape,
    up: vectorShape,
    width: Types.number,
    height: Types.number,
  };

  class TraceKernel extends KernelContext {
    @kernelFunction(Types.number, [vectorShape, vectorShape])
    vdot(v1: Vector3, v2: Vector3): number {
      return v1.x * v2.x + v1.y * v2.y + v1.z * v2.z;
    }

    @kernelFunction(vectorShape, [vectorShape, vectorShape])
    vcross(v1: Vector3, v2: Vector3): Vector3 {
      return {
        x: v1.y * v2.z - v1.z * v2.y,
        y: v1.x * v2.z - v1.z * v2.x,
        z: v1.x * v2.y - v1.y * v2.x,
      };
    }

    @kernelFunction(vectorShape, [vectorShape, vectorShape])
    vadd(v1: Vector3, v2: Vector3): Vector3 {
      return { x: v1.x + v2.x, y: v1.y + v2.y, z: v1.z + v2.z };
    }

    @kernelFunction(vectorShape, [vectorShape, 1])
    vmul(v: Vector3, l: number): Vector3 {
      return { x: v.x * l, y: v.y * l, z: v.z * l };
    }

    @kernelFunction(1, [vectorShape])
    vlen(v: Vector3): number {
      return this.sqrt(v.x * v.x + v.y * v.y + v.z * v.z);
    }

    @kernelFunction(colorShape, [colorShape, colorShape])
    cmul(c1: Color, c2: Color): Color {
      return {
        r: c1.r * c2.r,
        g: c1.g * c2.g,
        b: c1.b * c2.b,
      };
    }

    @kernelFunction(1, [vectorShape, vectorShape, vectorShape])
    tarea(v1: Vector3, v2: Vector3, v3: Vector3): number {
      const a = this.vlen(this.vadd(v2, this.vmul(v1, -1)));
      const b = this.vlen(this.vadd(v3, this.vmul(v1, -1)));
      const c = this.vlen(this.vadd(v3, this.vmul(v2, -1)));
      const p = (a + b + c) / 2;

      return this.sqrt(p * (p - a) * (p - b) * (p - c));
    }

    @kernelFunction(vectorShape, [vectorShape])
    vnormal(v: Vector3): Vector3 {
      const l = this.vlen(v);
      return { x: v.x / l, y: v.y / l, z: v.z / l };
    }

    @kernelFunction(vectorShape, [vectorShape, vectorShape, vectorShape, vectorShape])
    vchangeSpace(v0: Vector3, vx: Vector3, vy: Vector3, vz: Vector3): Vector3 {
      return this.vnormal({
        x: v0.x * vx.x + v0.y * vy.x + v0.z * vz.x,
        y: v0.x * vx.y + v0.y * vy.y + v0.z * vz.y,
        z: v0.x * vx.z + v0.y * vy.z + v0.z * vz.z,
      });
    }

    @kernelFunction(rayShape, [Types.number, Types.number, cameraShape])
    getPrimaryRay(x: number, y: number, cam: Camera): Ray {
      const camRight = this.vcross(cam.forward, cam.up);
      const ya = -((2.0 * y) / cam.height - 1.0);
      const xa = (2.0 * x) / cam.width - 1.0;
      return {
        origin: cam.position,
        direction: this.vnormal(this.vadd(this.vadd(this.vmul(cam.up, ya), this.vmul(camRight, xa)), cam.forward)),
      };
    }

    @kernelFunction({ dist: Types.number, matIndex: Types.number, normal: vectorShape }, [
      [triangleShape],
      lengthShape,
      [sphereShape],
      lengthShape,
      rayShape,
    ])
    getIntersection(
      objects: Triangle[],
      objectsSize: typeof lengthShape,
      spheres: Sphere[],
      spheresSize: typeof lengthShape,
      ray: Ray,
    ): { dist: number; matIndex: number; normal: Vector3 } {
      const epsilon5 = 0.00001;
      const epsilon3 = 0.001;
      let distToClosest = this.INFINITY;
      let closestMatIdx = -1;
      let normal = { x: 0, y: 0, z: 0 };

      for (let i = 0; i < objectsSize.length; i += 1) {
        const d = objects[i].d;
        const t = (-d - this.vdot(objects[i].normal, ray.origin)) / this.vdot(objects[i].normal, ray.direction);

        if (t > epsilon5 && t < this.INFINITY) {
          const ip = this.vadd(ray.origin, this.vmul(ray.direction, t));
          const abc = this.tarea(objects[i].position.a, objects[i].position.b, objects[i].position.c);
          const pbc = this.tarea(ip, objects[i].position.b, objects[i].position.c);
          const apc = this.tarea(objects[i].position.a, ip, objects[i].position.c);
          const abp = this.tarea(objects[i].position.a, objects[i].position.b, ip);
          if (pbc + apc + abp <= abc + epsilon3 && t < distToClosest) {
            distToClosest = t;
            closestMatIdx = objects[i].materialIdx;
            normal = this.copy(objects[i].normal);
          }
        }
      }

      for (let i = 0; i < spheresSize.length; i += 1) {
        const v = this.vadd(ray.origin, this.vmul(spheres[i].position, -1));
        const b = this.vdot(v, ray.direction);
        const d2 = b * b - this.vdot(v, v) + spheres[i].radius;
        if (d2 > epsilon5) {
          const d = this.sqrt(d2);
          const t1 = -b - d;
          const t2 = -b + d;

          if (t1 > epsilon5 && t1 < distToClosest) {
            const hitPoint = this.vadd(ray.origin, this.vmul(ray.direction, t1));
            distToClosest = t1;
            closestMatIdx = spheres[i].materialIdx;
            normal = this.vmul(this.vadd(hitPoint, this.vmul(spheres[i].position, -1)), 1 / spheres[i].radius);
          } else if (t2 > epsilon5 && t2 < distToClosest) {
            const hitPoint = this.vadd(ray.origin, this.vmul(ray.direction, t2));
            distToClosest = t2;
            closestMatIdx = spheres[i].materialIdx;
            normal = this.vmul(this.vadd(hitPoint, this.vmul(spheres[i].position, -1)), 1 / spheres[i].radius);
          }
        }
      }
      return { dist: distToClosest, matIndex: closestMatIdx, normal: normal };
    }

    @kernelFunction(Types.number, [Types.number])
    hash(x: number): number {
      let h = this.uint(x);
      h += h << 10;
      h ^= h >> 6;
      h += h << 3;
      h ^= h >> 11;
      h += h << 15;

      return 1.0 * h;
    }

    @kernelFunction(Types.number, [{ current: Types.number }])
    random(seed: { current: number }): number {
      const a = 1103515245;
      const c = 12345;
      const m = 2147483648;

      let r = a * seed.current + c;
      r = r - this.int(r / m) * m;
      seed.current = r;

      return r / m;
    }

    @kernelFunction(rayShape, [vectorShape, rayShape, vectorShape, Types.number, Types.number])
    scatter(hitPoint: Vector3, ray: Ray, surface: Vector3, rd: number, ad: number): Ray {
      const ex = this.vcross(this.vmul(ray.direction, -1), surface);
      const ez = this.vcross(ex, surface);

      const c = this.sqrt(1 - rd);
      const hsv = this.vnormal({
        x: c * this.cos(this.M_PI * 2 * ad),
        y: this.sqrt(rd),
        z: c * this.sin(this.M_PI * 2 * ad),
      });

      return { origin: this.copy(hitPoint), direction: this.vchangeSpace(hsv, ex, surface, ez) };
    }

    @kernelEntry([
      {
        type: 'Object',
        shapeObj: infoShape,
        readWrite: 'read',
      },
      {
        type: 'Object[]',
        shapeObj: [triangleShape],
        readWrite: 'read',
      },
      { type: 'Object', shapeObj: lengthShape, readWrite: 'read' },
      {
        type: 'Object[]',
        shapeObj: [sphereShape],
        readWrite: 'read',
      },
      { type: 'Object', shapeObj: lengthShape, readWrite: 'read' },
      {
        type: 'Object[]',
        shapeObj: [materialShape],
        readWrite: 'read',
      },
      {
        type: 'Object',
        shapeObj: cameraShape,
        readWrite: 'read',
      },
      {
        type: 'Float32Array',
        readWrite: 'write',
      },
    ])
    main(
      info: typeof infoShape,
      objects: Triangle[],
      objectsSize: typeof lengthShape,
      spheres: typeof sphereShape[],
      spheresSize: typeof lengthShape,
      materials: Material[],
      camera: Camera,
      out: Float32Array,
    ) {
      const maxDist = 0x100000;
      const imageSize = info.imageHeight * info.imageWidth;

      for (let i = this.get_global_id(0); i < imageSize; i += info.threads) {
        const seed = { current: this.hash(i) };
        const sumColor = { r: 0.0, g: 0.0, b: 0.0 };
        for (let it = 0; it < info.iterations; it += 1) {
          const stack = this.array(
            {
              ray: { direction: { x: 0, y: 0, z: 0 }, origin: { x: 0, y: 0, z: 0 } },
              color: materials[0].dr,
              depth: 0,
            },
            100,
          );

          let stackSize = 0;

          const x = 1.0 * (i % this.int(info.imageWidth)) - 0.5 + this.random(seed);
          const y = 1.0 * (i / info.imageWidth) - 0.5 + this.random(seed);

          const primaryRay: Ray = this.getPrimaryRay(x, y, camera);

          stack[stackSize] = { ray: this.copy(primaryRay), color: { r: 1.0, g: 1.0, b: 1.0 }, depth: 3 };
          stackSize += 1;

          for (; stackSize > 0; ) {
            const rayInfo = this.copy(stack[stackSize - 1]);
            stackSize -= 1;

            if (rayInfo.depth > 0) {
              const hit = this.getIntersection(objects, objectsSize, spheres, spheresSize, rayInfo.ray);
              if (hit.dist < maxDist) {
                const hitPoint = this.vadd(rayInfo.ray.origin, this.vmul(rayInfo.ray.direction, hit.dist));

                if (materials[hit.matIndex].e.r > 0) {
                  const c = this.cmul(rayInfo.color, materials[hit.matIndex].e);
                  sumColor.r += c.r;
                  sumColor.g += c.g;
                  sumColor.b += c.b;
                }

                if (materials[hit.matIndex].dr.r > 0) {
                  stack[stackSize] = {
                    ray: this.scatter(hitPoint, rayInfo.ray, hit.normal, this.random(seed), this.random(seed)),
                    color: this.cmul(materials[hit.matIndex].dr, rayInfo.color),
                    depth: rayInfo.depth - 1,
                  };
                  stackSize += 1;
                }

                if (rayInfo.depth == 1) {
                  sumColor.r += rayInfo.color.r * materials[hit.matIndex].dr.r * 0.05;
                  sumColor.g += rayInfo.color.g * materials[hit.matIndex].dr.g * 0.05;
                  sumColor.b += rayInfo.color.b * materials[hit.matIndex].dr.b * 0.05;
                }
              }
            } else {
            }
          }
        }
        sumColor.r /= info.iterations;
        sumColor.g /= info.iterations;
        sumColor.b /= info.iterations;

        if (sumColor.r > 1) {
          sumColor.r = 1;
        }
        if (sumColor.g > 1) {
          sumColor.g = 1;
        }
        if (sumColor.b > 1) {
          sumColor.b = 1;
        }
        out[3 * i] = sumColor.r;
        out[3 * i + 1] = sumColor.g;
        out[3 * i + 2] = sumColor.b;
      }
    }
  }

  const threads = 2000;
  const arr = new Float32Array(scene.width * scene.height * 3).fill(0xcc);
  try {
    const traceRays = gpgpu.createKernel(TraceKernel).setSize([threads], [10]);
    await traceRays(
      { threads, imageHeight: scene.height, imageWidth: scene.width, iterations },
      scene.objects,
      {
        length: scene.objects.length,
      },
      scene.spheres,
      {
        length: scene.spheres.length,
      },
      scene.materials,
      scene.camera,
      arr,
    );
  } catch (error) {
    console.log(gpgpu.getLastBuildInfo());
    console.log(error);
  }

  return arr;
}

export function saveToFile(image: Float32Array, file: string, width: number, height: number): void {
  const cs = createCanvas(width, height);
  const context = cs.getContext('2d');

  context.fillStyle = '#0ff';
  context.fillRect(0, 0, width, height);
  const imageData = context.getImageData(0, 0, width, height);
  const data = imageData.data;
  let iter = 0;
  for (let i = 0; i < data.length; i += 4) {
    data[i] = image[iter] * 255;
    data[i + 1] = image[iter + 1] * 255;
    data[i + 2] = image[iter + 2] * 255;
    iter += 3;
  }

  context.putImageData(imageData, 0, 0);

  writeFileSync(file, cs.toBuffer('image/png'));
}