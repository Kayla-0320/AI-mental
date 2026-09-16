import { PrismaClient } from '@prisma/client';
const p = new PrismaClient();
(async () => {
  const bookings = await p.expertBooking.findMany({
    include: {
      patient: { select: { id: true, nickname: true } },
      consultant: { select: { id: true, nickname: true } },
    },
    orderBy: { createdAt: 'desc' },
  });
  console.log('Bookings:', JSON.stringify(bookings, null, 2));

  const users = await p.user.findMany({
    where: { role: 'CONSULTANT' },
    select: { id: true, email: true, nickname: true },
  });
  console.log('\nConsultant Users:', JSON.stringify(users, null, 2));

  await p.$disconnect();
})();
